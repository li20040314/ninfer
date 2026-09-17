# RTX 4060 (sm_89) 9B 适配 · L2 收尾交付小结

日期：2026-09-16 · 范围：L2 收尾 + 全量算子回归 · 目标机：RTX 4060 Laptop（24 SM / 8 GB）

---

## 1. 结论

| 项目 | 结果 |
|---|---|
| L2 三算子（`attn_input_proj` / `gdn_input_proj` / `linear_swiglu` q4）9B 适配 | ✅ 完成 |
| sm_89 全量算子回归（12 个目标） | ✅ **11/12** |
| 新增/修复的真移植缺陷 | 2 个（Q5 slab 数烤死、合作启动网格越界） |
| 判据标定问题（非代码缺陷） | 1 个，已修正并写明推导 |
| 遗留（跨架构既有问题，未擅自改） | 1 个，见 §4 |

---

## 2. 本轮解决的三件事

### 2.1 `linear_swiglu` q4 9B 判据超标 1.3% —— 判据问题，**不是**代码缺陷

| 项 | 内容 |
|---|---|
| 现象 | `ninfer_linear_swiglu_q4_a16_test` 仅 9B `T=128 graph` 失败：rel_l2=0.0033429，上限 3.3e-3（ratio **1.013**） |
| 可疑点 | 同用例 eager 与 graph phase 0 均 0.927；27B 全过 → 不像几何适配错误 |

**定位手法（已固化为测试诊断，可复用）**

1. 查路由表：T=128 落在 `Materialized`（`linear` 出 bf16 gate/up + `silu_mul`）。
   该路由在 27B 上误差也偏高（ratio≈0.93），融合 MMA 路由只有 ≈0.49 —— 差值正是 **bf16 中间结果**。
2. 给 fp64 oracle 加"模拟 Materialized"模式：gate/up 先舍入 bf16 再算 `silu*up`，
   得到该路由的**理论最优输出**（环境变量 `NINFER_TEST_SWIGLU_MATERIALIZED_FLOOR=1`）。

| 比对 | 结果 |
|---|---|
| 内核 vs 理论地板（Materialized 区间 22 个用例） | rel_l2 最大 **8.3e-8**（≈0）⇒ **内核已逐位最优** |
| 理论地板 vs 精确解 | 27B 3.182e-3；9B 符号翻转 T=128 graph **3.343e-3** |

理论解释：`d(out)/out = [1 + g(1−σ(g))]·d(gate)/gate + d(up)/up`，灵敏度因子
`|1 + g(1−σ(g))|` 随 `g → −∞` **无界增长** ⇒ 该路由的误差地板**依赖数据分布**，
27B 与 9B 的夹具数据本就不同，地板不同是正常的。

**处理**：A16 `relative_l2` 3.3e-3 → **3.7e-3**（≈实测最高地板的 1.11×），注释写明推导与复测要求。
融合 MMA 路由实测仍 1.6e-3，判据对其保留 2× 余量 ⇒ **放宽不损失检测力**。
测试转 `OK`，最大 ratio 0.9035。

### 2.2 `gdn_gating_proj` 合作启动崩溃 —— **真移植 bug，已修**

| 项 | 内容 |
|---|---|
| 现象 | rc=3221226505，`cudaErrorCooperativeLaunchTooLarge: too many blocks in cooperative launch` |
| 根因 | `cooperative_resident_ctas_per_sm()` 返回**在 sm_120a 上标定的编译期常量**；sm_89 每 SM 只有 **1536** 线程（5090=2048）、**24** SM（5090=170），实际驻留数普遍低于标定值 |

实测（`cudaOccupancyMaxActiveBlocksPerMultiprocessor`）：

| split / 线程 / 动态 smem | 5090 标定 | sm_89 实测 | |
|---|---|---|---|
| 8 / 256 / 40K（27B） | 2 | 2 | 一致 |
| 4,2 / **512** / 40K（27B） | 2 | **1** | 标定偏高 |
| 32 / 256 / 24K（35B） | 2 | 2 | 一致 |
| 16 / 256 / 24K（35B） | 4 | 4 | 一致 |
| 8,4,2 / 256 / 24K（35B） | 4 | **3** | 标定偏高 |

**修法**（`src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu`）：
原常量改名 `qualified_resident_ctas_per_sm()` 作为**上界**；新增 `resident_ctas_per_sm()`
用占用率 API 查询并取 `min(查询值, 上界)`（Full 与 Predicated 两个实例化都查、取小）。
⇒ **能达到标定值的设备行为完全不变**，只有达不到的设备才分更多、更小的块。
崩溃消除后，684 个用例跑到只剩 1 个边界项。

### 2.3 全量回归

`deploycrt 89` 后运行（一键脚本 `.workbuddy/tmp/run_tests.py`）：

| 目标 | 结果 |
|---|---|
| `ninfer_attn_input_proj_test` | ✅ OK |
| `ninfer_gdn_input_proj_test` | ✅ OK |
| `ninfer_gdn_input_proj_conv_snapshot_test` | ✅ OK |
| `ninfer_gdn_input_proj_conv_record_test` | ✅ OK |
| `ninfer_gdn_gating_proj_test` | ❌ FAIL（见 §4，跨架构既有问题） |
| `ninfer_linear_{q4,q5,q6}_a16_test` | ✅ OK ×3 |
| `ninfer_linear_add_{q4,q5}_a16_test` | ✅ OK ×2 |
| `ninfer_linear_swiglu_q4_a16_test` | ✅ OK |
| `ninfer_device_test` | ✅ OK |

⚠️ **不要跑全量 `build 89`**：`tests/artifact/fixture.h` 的 `mkdtemp`、
`tests/test_context_cost.cpp` / `test_pretty_logging.cpp` 的 `<unistd.h>`
是**与本移植无关的既有 POSIX 依赖**，会直接失败。改算子只 `msvcbuild.py targets 89 <目标>`。

---

## 3. 本轮改动的文件

| 文件 | 改动 |
|---|---|
| `src/ops/linear_swiglu/q4/q4_linear_swiglu_geometry.h` | 新增（27B+9B 几何表，单一真源） |
| `src/ops/linear_swiglu/q4/q4_linear_swiglu_gemv.cu` | `Q4SwiGluProfile<GateUpRows, InputRows>` 参数化；分派改普通 `switch`（规避 nvcc 类类型 NTTP 的 host stub 限制） |
| `src/ops/linear_swiglu/q4/q4_linear_swiglu_plan.cpp` | `supported_shape` 改查几何表 |
| `src/ops/wrapper/linear_swiglu.cpp` | 加 9B 形状分支 |
| `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu` | 合作启动驻留 CTA 数改运行时占用率查询（§2.2） |
| `tests/ops/linear_swiglu/linear_swiglu_test_common.cpp` | A16 判据 3.3e-3→3.7e-3（含推导注释）；新增 env 门控的"Materialized 误差地板"诊断 |
| `tests/ops/linear_swiglu/test_q4_a16.cpp` | 新增 9B profile 用例 |
| `doc/rtx4060-sm89-9b-adaptation-plan.md` | §3 四层→五层（补 L5 `weight_input.cpp`）；§4.6 标注其 9B 分支现状 |

---

## 4. 遗留：1 项（非移植回归，待项目负责人定夺）

| 项 | 值 |
|---|---|
| 用例 | `ninfer_gdn_gating_proj_test` → `qwen3_6_27b T=4097 g` |
| 实测 | rel_l2 = **1.405e-6**，上限 `kGdnProjectionFp32` = 1.4e-6（超 **0.36%**）；gross_ratio 也已是 0.97 |
| 原因 | 路由表 **T 越大 SplitK 越小**（8→4→2→1），T=4097 走 `MmaUnsplit`，单 CTA 在 fp32 累加全部 5120 项 → 误差随 T 单调增大：1.1e-7@T=1024 → 2.8e-7@1025 → 4.5e-7@2049 → **1.1e-6@4097**，与 `√K·ε_fp32` 量级完全吻合 = 纯 fp32 累积地板 |
| 为何与 sm_89 无关 | ① 该路径 `SplitK==1`，**不读 `multiprocessor_count`、不做分块**（无设备相关分支）；② `mma.cuh` 里**只有 `mma_fp8_e4m3`** 有 `__CUDA_ARCH__>=1000` 分支，`mma_bf16` 跨架构一致 ⇒ **跨架构必现**，只是此前被 §2.2 的崩溃掩盖 |
| 建议 | 若认可"判据应覆盖最差路由"，把 `kGdnProjectionFp32.relative_l2` 由 1.4e-6 提到 ≈1.6e-6。**未擅自修改**：该判据是 fp32 精确性闸门，且 9B 不走此几何（属 27B/35B 路径） |

---

## 5. 下一步（S1 剩余）

1. **L3**：`src/ops/wrapper/gdn_gating_proj.cpp` 加 9B（`parent.n == 64 && parent.k == 4096`），
   并同步 `src/ops/weight_input.cpp:202` 的 `prepare_gdn_gating_proj_weights()`（`(48,5120)‖(32,2048)` → 追加 `(64,4096)`）。
2. **Task D**：Qwen3.5-9B 的 recipe（`tools/convert/qwen3_5.py` + `official_recipes.py`）。
3. **Task F**：4060 真机冒烟（`--components text --kv-dtype int8 --max-context 4096`）。
4. 提交当前 ~66 个改动文件。
