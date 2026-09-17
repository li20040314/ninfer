# Phase 1 交付：CPU RowSplit GEMM 内核 + 引擎接线

> 目标：在 CPU 上做 RowSplit 量化权重的 GEMV/GEMM，作为 offload 场景下 PCIe 流送的替代路径。
> 状态：**内核正确性已锁死；性能显著优于 PCIe；已完成引擎接线（Phase 1b）并通过端到端一致性验证**。
> 日期：2026-09-17　　目标架构：sm_89（RTX 4060 Laptop 8GB）+ i9-13900H（6P+8E / 20 逻辑核）+ 2×16GB DDR5-5200

---

## 1. 交付物

| 文件 | 作用 |
|---|---|
| `src/ops/cpu/rowsplit_gemm.h` | 公开接口：`rowsplit_gemm_supported` / `rowsplit_gemm` / `rowsplit_worker_threads` / `set_rowsplit_worker_threads` / `rowsplit_avx2_available` |
| `src/ops/cpu/rowsplit_gemm_detail.h` | 内部契约：`RowSplitGeometry`、`gemm_rows` / `gemm_rows_scalar` 声明、精确的 half/bf16 转换 |
| `src/ops/cpu/rowsplit_gemm.cpp` | 几何派生（以引擎 `weight_geometry()` 为唯一真源）、扁平抢单线程池、激活 staging |
| `src/ops/cpu/rowsplit_gemm_avx2.cpp` | AVX2 内核（单独 TU，`/arch:AVX2`），入口检查 `host_has_avx2()` |
| `src/ops/cpu/rowsplit_gemm_scalar.cpp` | 标量参考；q6 与非 AVX2 主机走此路径 |
| `tests/ops/cpu/test_rowsplit_gemm_cpu.cpp` | 一致性测试（目标 `ninfer_linear_cpu_rowsplit_test`） |
| `src/core/platform.{h,cpp}` | 新增 `physical_core_count()`：Win32 用 `RelationProcessorCore` 数**物理**核，避开 SMT 逻辑核翻倍 |

覆盖量化格式：**Q4_G64_FP16 / Q5_G64_FP16 / Q6_G64_FP16 / Q8_G32_FP16**。

---

## 2. 契约要点

- **几何来自引擎自身**：`derive_geometry()` 内部调用 `weight_geometry(qtype, layout, shape)` 派生子几何，
  CPU 侧不与设备侧布局漂移；`padded_shape[1]` 与父对象不一致时直接拒绝（否则会静默读错相邻行）。
- **q4/q8 没有 high 平面**（q4 是整半字节、q8 是整字节）⇒ 只有 5/6 位格式要求 `qhigh` 非空——初版在这里误拒过。
- **激活 staging 只做转置**：设备侧给 `[k][tokens]`，CPU 侧要 `[tokens][padded]`；**不做任何置换**（见 §3.3）。
- **线程池**：单例、扁平抢单、一次栅栏（每层一次调用 ⇒ 唤醒开销可摊薄）；线程数取**物理核数−1**。

---

## 3. 正确性门禁（已通过）

### 3.1 测试矩阵
- 5 个几何：`8×1000`（K 非 128 倍数，验证 padding 列严格贡献 0）、`37×5120`、`64×512`、`128×17408`、`256×5120`
- 6 个 token 数：1, 2, 3, 4, 5, 8
- 4 种格式 × 上述全部组合
- 判据：A16 契约 `rel_l2 ≤ 1/256` 且 `max_abs ≤ gross_absolute + gross_rel_to_max_ref × max_reference`
- oracle：用测试夹具自己解码出的**精确**权重做 fp64 点积（不复用内核的算术）

结果：**全部通过，无 `first_non_finite`**。

### 3.2 拒绝用例（全部抛 `std::invalid_argument`）
无 code 平面 / `Contiguous` 布局 / `tokens == 0` / 输出为 null / **padded K 与父对象不一致**。

### 3.3 三处真实缺陷（初版，非笔误）

| # | 问题 | 根因 | 修复 |
|---|---|---|---|
| 1 | ❌ q4 被误拒 | 代码假设"非 8 位就必须有 high 平面" | 只有 `bits ∈ {5,6}` 才要求 `qhigh` |
| 2 | ❌ 判据误报（t=1 时 max_abs 超标） | 拿 `max_abs` 单独比 `gross_absolute`；实际 limit 含 `gross_rel_to_max_ref × max_reference` 项，t=1 时该参考值小 | 改用 `reduction_passes()` + `gross_error_limit()` |
| 3 | ❌ Q5 `rel_l2` 0.76~1.3（形状全对） | 移植了微基准的**"块内 4 字节 + 激活置换 `[0,4,1,5,2,6,3,7]`"** 约定 | 引擎/夹具是**"相邻对"约定**（字节 j 低半字节 = 元素 2j），该约定下 `unpacklo/unpackhi` 天然产出元素自然序 ⇒ **删除全部激活置换** |

> ★ 第 3 条是本节最有价值的沉淀：**两套位布局约定长得几乎一样，错了只在 `rel_l2` 上暴露**。
> 判定锚点：引擎 `q4_rowsplit_storage.cuh` 的 `decode_pair`（`q0 = packed & 0x0F`）与夹具 `unpack_lowbit_code`（low/high 交替）。

---

## 4. 性能：三次迭代与两个根因

### 4.1 迭代轨迹（q8 8192×5120）

| 版本 | t=4 @13 线程 |
|---|---|
| 初版：栈上 `chunks[8]` + 标量 `half_to_float` | 16.9 GB/s |
| 去栈数组（全部寄存器化） | 22.6 GB/s |
| **+ 多累加器 + F16C** | t=1 达 **29~72 GB/s**（随系统状态，见 §6） |

### 4.2 两个根因（可复用）

**① 栈数组的 store-to-load forwarding 停顿**
`chunks[8]` 里每个元素由 16 字节 xmm store 写入，随后 `_mm256_cvtepi8_epi32(chunks[i])` 按 **8 字节**读回——
宽度不匹配，store buffer 无法转发，每组 8 次转发停顿。**把中间值全部留在寄存器**（手写展开而不是数组）即 **+34%**。

**② 串行 FMA 依赖链**
单一 `group_accumulator` 把 8 次 `contract8` 串成一条链：8 × 4 cycle = **32 cycle/组**，
而独立 uops 只需 **8 cycle**。拆成 **q8 两条链 / q4·q5 四条链**后 t=1 提升约 **3×**。

**③ 附带**：`scale` 转换改用 **F16C `_mm_cvtph_ps`**，替掉标量 `half_to_float` 的 if/else 分支链。

### 4.3 瓶颈归属（**当日修正：算力受限，与 tokens 无关**）

⚠️ 本节初版结论（"t=1 内存受限、t=4 算力受限"）**是错的**，在补测 GFLOP/s 后推翻。留此存档以免重犯。

初版依据的是字节吞吐随 tokens 崩塌（72 → 19 GB/s）。但**字节吞吐不是不变量**：q8 每元素一字节，
t=1 时每字节权重只做 2 FLOP，t=4 时做 8 FLOP。同一份算力，tokens 越大，折算出的字节数越少。
把 FLOP 还原出来：

| tokens | q8 GFLOP/s | q4 GFLOP/s |
|---|---|---|
| 1 | 136 | 165 |
| 4 | 141 | 157 |
| 16 | **150** | 141 |
| 64 | **145** | 138 |

**恒定在 138~165 GFLOP/s，与 tokens 无关 ⇒ 内核自始至终是算力受限**，从来没有吃到 DRAM 带宽。

⚠️ 连带更正两条被它污染的结论：

1. "decode 形状已吃满 DRAM ⇒ Phase 3 预期上调" —— **不成立**。DAX 带宽从来不是上限，算力才是。
2. "CPU 路径可与 MTP 正交叠乘" —— **不成立，且方向相反**。MTP 提高的是每轮产出，却不降低每 token 的
   算力需求：一轮要跑 1 次 target(1 token) + 1 次 verify(4 token) + 3 次 draft 的 lm_head，约 250 GFLOP
   换 2.04 个 token ≈ **123 GFLOP/token**，是纯 decode（50 GFLOP/token）的 **2.4×**。
   在算力受限的 CPU 上，MTP 会**变慢 2.4 倍**（848 ms/token vs 345 ms/token）。

**Phase 3 的量化基线**：内核 **≈145 GFLOP/s**；27B 每 token ≈50 GFLOP ⇒ 纯 CPU 前向 **≈345 ms/token ≈ 2.9 tok/s**
（当前 PCIe 路径 ≈1000 ms/token）。**前提是关掉 MTP**——这正是 §4.3 更正后 Phase 3 计划的形态。

---

## 5. 与 PCIe 的对比

| 路径 | 有效带宽 | 相对 |
|---|---|---|
| 现状 PCIe 流送 | ~13 GB/s（历史实测） | 1.0× |
| CPU GEMV（本次，t=1） | 29~72 GB/s | **2.2× ~ 5.5×** |

⚠️ 取自不同批次的极值区间，仅表数量级；端到端收益以 §7 的 27B 基准为准。

---

## 6. 测量方法学与干扰（重要）

同机不同时段测同一份二进制，结果可差 **2~3 倍**：

| 指标 | 充裕时段 | 紧张时段 |
|---|---|---|
| 访存探针（768 MiB 顺序读，14 线程） | 58.4 GB/s | 50.3 GB/s（−14%） |
| 访存探针（单线程） | 23.2 GB/s | 24.2 GB/s（持平） |
| **内核 q8 t=1（单线程）** | **11.4 GB/s** | **6.0 GB/s（−47%）** |
| **内核 q8 t=1（13 线程）** | **71.7 GB/s** | **29.0 GB/s（−60%）** |

- 访存受限的探针几乎不动，**含计算的内核大幅漂移** ⇒ 与 CPU 频率/调度/内存压力相关，不是内核问题。
- 观测到的条件：可用物理内存仅 **2.84 / 33.3 GB**，`vmware-vmx`、`idea64`、`WorkBuddy` 常驻。
- ✅ **结论与纪律**：绝对性能**只信端到端 27B 基准**；内核数据只用**同批次内自洽的相对比较**（扩展性、q4↔q8、t=1↔t=4）。
- 工具：`.workbuddy/tmp/bw_probe.cpp`（独立带宽探针）、`.workbuddy/tmp/runtest.py`（可靠地跑 exe 并回显）。

---

## 7. 踩坑清单（写代码前必读）

1. ★★ **静态析构期的崩溃会长得像内核崩**：`RowSplitWorkers` 单例的 `std::vector<std::thread>` 在进程退出时仍有 12 个 joinable worker，
   `~thread()` 直接 `std::terminate` → **0xC0000409**，发生在 `main` 返回**之后**。
   症状极具误导性：日志内容完全正确、退出码却是 FastFail；叠加 stdout 全缓冲后变成"全无输出 + 409"，看起来像内核段错误。
   修法：`~RowSplitWorkers()` 里置 `stopping_` + `notify_all()`，**锁外** join。
2. **测试必须无缓冲**：`std::setvbuf(stdout, nullptr, _IONBF, 0)`，否则崩溃前的输出全部丢失。
3. **PowerShell 的 `*>` 重定向在本机不可靠**（写 UTF-16，且有时根本不替换目标文件 → 会读到上一次的数字当真）。
   改用 `.workbuddy/tmp/runtest.py`（Python `subprocess` 捕获 + `print`）。
4. `msvcbuild.py targets` 的日志在 `D:\deps\build\targets-<arch>.log`，不是重定向文件。
5. 测试进程自身的大分配（如为参照条目 pack 一个 `n×k` 的 `dequant`）会叠加到本就紧张的系统内存上，进一步污染测量。

---

## 8. Phase 1b：接线（已完成）

### 8.1 设计：按指针位置自动分流，不新增任何声明

原本设想要给 `Weight` 加 `residency` 字段（要动 `weight.h` / `weight_view` / `prepare.cpp` / 模型层共 5~6 处）。
实际上**不需要**：

- `bind_view()`（`src/artifact/views.cpp:14`）**已经**按 `Residency` 选择 `host_parent()` 或 `device_parent()`；
  这正是 `--host-embedding` 的机制。所以"host 权重"到达 ops 层时，`Weight::qdata` **本来就是 host 指针**。
- 缺的只是**执行层不知道该走 CPU**。而 UVA 下 host/device 指针可由 CUDA 自己区分。

于是接入点只有一处：`dispatch_linear()` 开头一次探测。

| 指针来源 | `cudaPointerGetAttributes` 结果 | 路由 |
|---|---|---|
| 设备分配 | 成功，`cudaMemoryTypeDevice` / `Managed` / `Array` | 原有设备内核（**不变**） |
| CUDA 注册的 host 内存 | 成功，`cudaMemoryTypeHost` | CPU 收缩 |
| 普通 host 内存（pageable） | 成功，**`cudaMemoryTypeUnregistered`**（现代运行时）或不解析（旧运行时） | CPU 收缩 |
| 其它未知类型 | — | 保守留在设备路由（读错方向会 fault，不能赌） |

⚠️ **踩坑**：最初只把 `cudaMemoryTypeHost` 当 host，结果**所有用例都被判成 device**——
现代 CUDA 对"未注册的普通 host 内存"返回的是 `cudaMemoryTypeUnregistered`，**不是**查询失败。
判定必须显式枚举（见上表）。

**缓存**：单槽记录上次的 `(指针 → 判定)`。decode 循环每 token 走同一批权重、指针稳定，命中率高，
于是每次 linear 调用的探测退化为每个权重一次。

### 8.2 数据流

```
device 激活 ──cudaMemcpyAsync(D2H)──> thread_local 暂存 ──> rowsplit_gemm() ──> thread_local 输出
                                                                                    │
                                                    cudaMemcpyAsync(H2D，不必同步)<──┘
```

- 激活 `cudaStreamSynchronize` 是必需的：CPU 随后要读它，且暂存缓冲会被下一层复用。
- 输出回写**保持异步**：同 stream 上的后续 kernel 天然有序，CPU 不必等自己刚算出的结果。
- 每层一次跨总线：`2·K·T` 字节。T=1 时几十 KB，对上被替代掉的 MB 级权重流送，代价可忽略。

### 8.3 交付物与验证

| 文件 | 作用 |
|---|---|
| `src/ops/linear/host/linear_host.h` | `weight_is_host_resident()` / `linear_host()` |
| `src/ops/linear/host/linear_host.cu` | 探测（含单槽缓存）+ D2H/CPU/H2D 编排 |
| `src/ops/linear/linear.cpp` | `dispatch_linear()` 开头加分流（唯一改动的既有文件） |
| `tests/ops/linear/test_linear_host.cpp` | 端到端一致性（目标 `ninfer_linear_host_cpu_test`） |

**验证方式刻意选在公共入口**：测试通过 `ninfer::ops::linear()` 调用，
**不告诉引擎权重在哪**——若没有自动分流，设备内核会读到 host 指针而 fault 或产出乱值，fp64 oracle 会抓住。

结果：**Q4/Q5/Q6/Q8 × 3 几何（64×512 / 37×5120 / 128×17408）× 3 token 数（1/3/8）全部通过 A16 判据**，
外加一条**双向探测断言**：host 权重必须判为 host、device 权重必须判为 device、且缓存必须被替换（不得残留上一次的判定）。

### 8.4 遗留

- ✅ 未回归破坏设备路径：探测对 device 指针返回 false 直接走原路径，逻辑上不可能改变形状可用性；
  实测 `ninfer_linear_q8_a16_test` 仍是基线里那个 sm_89 形状门控失败（`n2048 k4096` 静态共享内存超限），**非本次引入**。
  完整回归套件（12 套件）待下一轮跑。
- ✅ 已做（§9）：offload plan 能把指定 linear 标为 `Residency::Host`（`--host-output-head`），并跑了端到端 27B 基准。
- ⏭ 还没做：Phase 3（全量 host linear）与 GPU 侧单 token 时间测量。

---

# Phase 1c：接进真实引擎（`--host-output-head`）

目标：让 CPU 收缩真正在 27B 上跑起来，验证"host 权重 → CPU 计算 → 回写 bf16"整条链路，
并为 Phase 3 拿到量化基线。

## 9.1 为什么第一个接 output head

`plan_weight_offload` 的注释早就点名了它：

> The output head is deliberately not on this list: it needs the whole 1.26 GiB every step, and
> zero-copy reads run at ~8.3 GB/s against 13 GB/s for a device upload. **It would need a CPU GEMV,
> which is a separate piece of work.**

那件 "separate piece of work" 就是 §1–§8 做完的事。选它的理由：

1. **格式落在覆盖面内**：`text/output_head` = `Q8_G32_FP16` RowSplit，`[248320, 5120]` = 1.26 GiB。
2. **永远是 t=1**，没有 GEMM 灾难：prefill 也只算最后一个 token（`Tensor last_xf = xf.slice(1, len - 1, 1)`）。
3. **它是唯一一个"必须常驻且每步全量读"的权重**，因此 CPU 化的收益最直接（省下的是常驻显存）。

## 9.2 接线：9 处改动，零新增机制

不需要给 `Weight` 加 residency 字段——`bind_view()` 本来就按 residency 选 host/device parent
（`--host-embedding` 用的就是这条路径）。新增的只是一个开关和一次归类。

| 文件 | 改动 |
|---|---|
| `include/ninfer/types.h` | `EngineOptions::host_output_head` |
| `src/models/load_options.h` | `LoadOptions::host_output_head` + `load_options()` 映射 |
| `src/models/qwen3_5/load.cpp` | `place_on_host(id, page_locked)` 归并两种 host 用途 |
| `apps/cli/options.{h,cpp}`、`main.cpp` | `--host-output-head` |
| `src/serve/serve_options.{h,cpp}`、`generation_service.cpp` | 同上（含 help 文本） |

**关键设计：host 权重也计入 `target_resident`。**

```cpp
std::map<std::size_t, bool> host_objects;   // object index -> page_locked
const auto place_on_host = [&](WeightId id, bool page_locked) {
    pending[id.index].reference.residency = artifact::Residency::Host;
    for (const auto& part : reference.binding.parts) { ...host_only_bytes += bytes; }
};
if (options.host_embedding)   { place_on_host(text.token_embedding, /*page_locked=*/true); }
if (options.host_output_head) { place_on_host(text.output_head,    /*page_locked=*/false); }
```

`target_resident = total_layer_bytes * (1 - ratio) + host_only_bytes`，所以腾出的 1.26 GiB
**自动变成更多常驻层**，少流送的正是这部分。两种 host 用途的 `page_locked` 必须分开：

- embedding 由 **device kernel 直接解引用** → 必须 page-locked；
- output head 由 **CPU 收缩**读取 → 只要可寻址，`page_locked=false` 走普通 host 拷贝，省掉一次 pinning。

## 9.3 ★★ 布局缺陷：t > 1 时输入与输出索引都反了

**症状**（只有真实引擎能暴露它）：开 MTP 时输出变成垃圾 `东风 일 Medizintoupper,to/get`，
**接受率 35.3% → 0.0%**、`acceptance length 2.04 → 1.00 tok/round`；关 MTP 时却完全正确。

**根因**：实现按 `[k][tokens]` 索引，引擎是 `[tokens][k]`（token 外层、k 内层）。

| 位置 | 错误写法 | 正确写法 |
|---|---|---|
| `stage_activation` 输入侧 | `activation[column * tokens + token]` | `activation[token * k + column]` |
| AVX2 / 标量内核输出侧 | `out[row * tokens + token]` | `out[token * out_row_stride + row]` |

内核算术本身没错（`x + token * padded_columns` 早就是对的），错的只是两侧的**元素寻址**。

**为什么两条测试都没抓到**：测试构造输入时用了**同一个公式**，与实现共享了错误假设 ——
自洽的闭环，`t=1` 到 `t=8` 全绿。而 `tokens == 1` 时两种顺序恰好重合，
所以 decode（t=1）与 prefill（t=1）全都正确，只有 MTP 的 verify（t=4）踩中。

**权威锚点**（认布局只认设备内核）：

- `bf16_n256_k5120.cuh`：`&x_shared[token * kGroupK + ...]`
- `fp8_a16_ksplit_mma.cuh`：`&x_shared[warp][token * kTileK + ...]`

**修正**：三处索引统一 + `gemm_rows`/`gemm_rows_scalar` 增加 `out_row_stride` 参数（= `weight.n`），
两个测试的 `make_activation` 与 oracle 一并改为权威顺序，并新增 `activation_at()` 帮助函数把约定集中在一处。

⚠️ **这是 Phase 3 的硬前提**：所有 streamed 层的 linear 都要在 CPU 上跑，
而 verify 每轮都是 `t=4`。这个 bug 不修，Phase 3 会以"接受率归零 + 输出乱码"的形式失败。

## 9.4 端到端数据（同批次，48 token）

| 配置 | decode tok/s | prefill tok/s | GPU weights | elapsed | 接受率 |
|---|---|---|---|---|---|
| 无 MTP，基线 | 1.0 | 7.9 | 3.80 GiB | 49.0 s | — |
| 无 MTP，**+host-output-head** | **1.2（+20%）** | 8.2 | 3.91 GiB | **44.0 s（−10%）** | — |
| MTP，基线 | 2.1 | 8.9 | 4.22 GiB | 26.4 s | 35.3% / 2.04 |
| MTP，**+host-output-head** | 2.0（−5%） | 8.5 | 4.33 GiB | 27.3 s | **35.3% / 2.04** |

**正确性零损失**：MTP 接受率与 acceptance length 与基线**逐位相同**，生成文本逐字相同。

**机制确证**（debug 级日志，关 MTP）：

| | device objects | host objects | H2D |
|---|---|---|---|
| 基线 | 159 | 616 | 3.80 GiB |
| +host-output-head | **241（+82）** | **534（−82）** | 3.91 GiB |

head 加了 1 个 host object，同时 **83 个层对象从 host 迁到 device**（多驻留），净 −82。
GPU 用量 +0.11 GiB 而非 −1.26 GiB 正是这个交换的签名：**释放的显存被拿去换常驻层了**。

**为什么 MTP 下没有收益**：CPU 侧一轮要算 5 次 head（3 draft + 1 verify(t=4) + 1 target），
约 6.3 GB 权重 @ ~60 GB/s ≈ 105 ms；而省下的流送 ≈ 1.3 GB / 13 GB/s ≈ 100 ms。**正好抵消**
（与 §4.3 更正后的算力分析一致）。

## 9.5 结论与用法

- `--host-output-head` 的价值：**无 MTP 时 +20%**（1.0 → 1.2 tok/s，模型耗时 −10%）。
- **MTP 开启时不要用**：净零（实测 2.0 vs 2.1，噪声内）。
- 真正的价值是**验证了整条链路**并**暴露了 t>1 的布局缺陷** —— 后者是 Phase 3 的硬前提。
- 用法：`run-27b.ps1 bench -NoSpec -MaxNum 48 -ExtraArgs '--host-embedding','--host-output-head'`

## 9.6 Phase 3 待办

1. 让 streamed 层也走 CPU：把 `WeightOffloadSpec` 的流送改成"不上传、由 CPU 收缩"，
   `ensure_layer()` 变为 no-op，`bind_view` 选 host parent。
2. **需要 host 化的 op**：`ops::linear` ✅ 已完成；`ops::linear_add`（down / mixer output）
   与 `ops::weight_input`（融合 qkv / gate-up，多父配对形态）**尚未支持**，各需独立实现 + 测试。
3. **必须关 MTP**（§4.3）。
4. 预期：**≈345 ms/token ≈ 2.9 tok/s**（当前 1000 ms/token）；实测后重估。
5. 顺带：内核 145 GFLOP/s 仅为 AVX2 峰值的 ~12%，优化空间大，但不阻塞 Phase 3。
