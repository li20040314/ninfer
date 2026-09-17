# L3 交付小结：gdn_gating_proj 与 linear_add Q5 的 9B 适配（2026-09-16）

前置：L2 交付小结（`rtx4060-sm89-l2-completion-2026-09-16.md`）。本轮完成精确问题准入（exact
problem）的 L3 wrapper 层与 L2 plan/内核层的 9B 最后一档几何，并补 Task D 的转换 recipe。

## 1. gdn_gating_proj 9B（heads=32, hidden=4096）

### 根因
MMA 内核完全由 `Geometry` 模板驱动（`kHidden` 编译期 → KTiles → SplitK static_assert），
plan 只认 `is_27`(48,5120) / `is_35`(32,2048)，9B `(32,4096)` 被拒。

### 修复（全部为新增，不动 27B/35B 既有实例）
| 文件 | 改动 |
|---|---|
| `bf16_gdn_gating_proj_gemm_mma.cuh` | 新增 `Bf16Gdn9Geometry{32, 4096, 64}`——**与 35B 同 CTA 形状**（BN64、2 head-tiles、8 warps、24KB smem），K 深度翻倍；寄存器/smem 资源 1:1 沿用 |
| `bf16_gdn_gating_proj_kernels.cu` | `require_shape9`；`qualified_resident_ctas_per_sm` 加 9B 分支（SplitK 16/8/4/2 → 上界 4）；5 个 launch 包装（unsplit/split16/8/4/2，全部 `Warps=8`） |
| `bf16_gdn_gating_proj_kernels.h` | 5 个声明 |
| `bf16_gdn_gating_proj_plan.cpp` | `is_9`；`k9Routes`（照 35B 五档：t≤127→split16、≤1024→split8、≤2048→split4、≤4096→split2、≥4097→unsplit；**无 4060 性能数据，边界待真机调优**）；`candidate_is_legal` 9B 分支（SIMT 内核烤死 k35K=2048 → 拒绝）；`mma_tile_cols` 9B→64；`execute_resolved` 四处三分支；resolve/capacity 三分支 |
| `wrapper/gdn_gating_proj.cpp` | `require_bf16_parent` 加 `(64, 4096)` → `{input_rows=4096, heads=32}` |
| `tests/ops/test_gdn_gating_proj.cpp` | `kQwen359` 几何：capacity 契约、投影 T∈{1,127,128,1024,1025,2049,4097}（覆盖全部 5 条新路由 + Full/Predicated token variant）、norm Composed T∈{1,6,16,17,64,129,1024,2048,4097} + capture/replay T=8 |

### 验证
- 685 用例仅既有 `qwen3_6_27b T=4097 g` 超标 0.36%（跨架构 fp32 累积地板，故意遗留）。
- 9B 全部新用例静默通过（含 workspace peak 断言、输入不可变、guard 区）。

## 2. linear_add Q5 9B（rows=4096；k=4096 attn/GDN output、k=12288 MLP down）

排查全量 wrapper 期间发现：9B text 的输出侧投影（Q5）走 `linear_add`，其闭表只有
`(5120,6144)`/`(5120,17408)`——**此缺口此前未被任何文档标注**（计划文档 §4.4 有预告但状态为待办）。

### 关键判断
- **MMA 内核对 K 完全运行时参数化**（`q5_rowsplit_gemm_mma_kernel` 收 rows/k/cols/padded_k
  运行时值；schedule 的 S=PipelineStages 与 K 无关，smem 预算 48KB 与 K 无关）→ 零内核改动。
- GEMV 模板约束 `K % 1024 == 0` → 4096/12288 均满足，加实例化即可。
- SIMT split2 的 `FullSlabs = k/1024`、`Stride = k`，直接扩表。

### 修复
| 文件 | 改动 |
|---|---|
| `q5_linear_add_plan.cpp` | `kSupports` 加两行；`kK4096Routes`（照 6144）/`kK12288Routes`（照 17408）；resolve_plan 二元硬切 → 四路 switch |
| `q5_linear_add_gemv.cu` | `<4096,4096,16,2,true>`（照 6144 用 staged-x）与 `<4096,12288,16,2,false>`（照 17408 不 stage） |
| `q5_linear_add_gemm_simt.cu` | `dispatch_shape` 加 k=4096→`<Cols,4,4096>`、k=12288→`<Cols,12,12288>` |
| `wrapper/linear_add.cpp` | Q5 supported_shape 加 `(4096,4096)`、`(4096,12288)` |
| `tests/ops/linear_add/test_q5_a16.cpp` | 两个新形状：区间起点 b-1/b/b+1 + 内点全套（seed 417/425） |

**q4_linear_add 经核实 S1 不需要**：9B 的 Q4 只用于输入侧投影（input proj / swiglu），
无 residual-add 路由（修正了计划文档原先"q4 是唯一烤死 (n,k) 的算子族"的担忧范围）。

### 验证
`ninfer_linear_add_q5_a16_test` Passed（4 个形状全路由）。

## 3. 确认为"无需改动"的项

| 项 | 结论 |
|---|---|
| `weight_input.cpp` L202 | `(32,4096)` 分支**本已存在**（前序工作加的），L5 全部完成 |
| `launcher/rmsnorm.cu` | d=4096 命中 `512≤d≤8192 && d%1024==0` 通用分支（512 线程 CTA + prefetch），5120 只是特化快路径 |
| `launcher/embed_gather.cu` | 9B embedding 走 Q6 通用 kernel（d/T 运行时），5120 特例仅 Q8 packed 路径 |
| `wrapper/linear_swiglu.cpp` | 9B Q4 形状（gate_up 24576×4096）前序已加 |
| `causal_conv1d_silu.cpp:162` | `x=8192 → out(2048,2048,4096)` 恰为 9B GDN conv（kg=16×128、vg=32×128）已注册 |
| `gdn_projected_conv.cu:123` | 9B 形状已注册 |

## 4. Task D：转换 recipe

`tools/convert/official_recipes.py`：新增 `qwen3_5_9b` = `_dense_groupwise(model, recipe, Q6)`
——`_dense_groupwise` 是纯名字模式驱动（几何无关），9B Dense 与 27B 共享同一 Q4/Q5/Q6 分配
（embedding/output_head→Q6；attn·gdn query/key、mlp gate/up→Q4；其余→Q5），全程绕开 sm_89
被剔除的 Q8。`qwen3_5.py` 架构适配器本身完全 config 驱动，无几何硬编码。py_compile 通过。

## 5. 遗留与下一步

| 项 | 说明 |
|---|---|
| **kv_cache 9B 适配** | `kv_cache_append.cpp` 的 `kKVHeads=8`/`kHeadDim=128` 编译期烤死；cyclic 内核 `static_assert(Capacity==2048\|\|4096)`。9B 为 kv_heads=4（+head_dim 256 全宽）。需先弄清 lanes 语义（k 形状 `[128,...]` 与 head_dim 256 的关系），是独立工作块。**真机冒烟前的最后代码缺口** |
| 权重下载 | HF 缓存仅有 `turboderp/Qwen3.5-9B-exl3` 的 refs（1KB，无权重）；需下载官方 safetensors（~18GB bf16，建议 `HF_ENDPOINT=https://hf-mirror.com`） |
| `linear_pair.cpp:96` | `first_weight.k` 闭表缺 4096；9B 关闭 dflash/proposal 时预计不触发，启用前必须补 |
| 性能调优 | 9B 所有路由边界均为照抄近邻 K 的起步值，无 4060 实测数据 |
| gating fp32 判据 | 27B T=4097 的 1.6e-6 建议值仍待负责人决定 |

## 6. 回归

12 套件 **11/12**，与基线一致（唯一 FAIL = 既有 27B T=4097 fp32 地板边界）。
9B 新增用例全部通过，27B/35B 既有路径零破坏。
