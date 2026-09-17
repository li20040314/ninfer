# NInfer 支持 RTX 4060 Laptop（sm_89 / 8GB）解决方案

> 目标：让 NInfer 在用户的 **NVIDIA GeForce RTX 4060 Laptop GPU（compute capability 8.9，
> 8188 MiB 显存，驱动 596.36）** 上编译并运行。
> 现状：NInfer 目前硬性绑定 `sm_120a`（RTX 5090 / Blackwell）。

---

## 0. 结论速览

**可行，改造成本中等（预计 1～2 天，主要是构建解耦 + 权重重转）。**

核心四件事（第 4 项由实测复核新增，见 §1.5）：

1. **剥离 NVFP4**：NVFP4 与 TMA 是 Hopper/Blackwell 专属特性，sm_89（Ada）既没有 TMA
   也没有 NVFP4。构建时必须把 NVFP4 内核从目标中排除，改用 **FP8 / INT4 / INT8 / BF16** 路径。
   KV cache 的 `Fp8KeyNvfp4Value`（k8v4=k8v4）同样依赖 NVFP4，一并排除。
2. **修复 FP8 mma 形式**：`ops/common/mma.cuh` 的 FP8 mma 用了 Blackwell 专属的
   `kind::f8f6f4` 形式，sm_89 编译直接报错。需按架构改用经典的
   `mma.sync...e4m3.e4m3.f32`（实测该形式在 sm_89/90/100a/120a **全部**通过）。
3. **换权重格式**：发布的模型卡全是 NVFP4（Blackwell）。转换工具本身已支持 `fp8_row`
   和 `groupwise(Q4/Q5/Q6/Q8)`，对应 `mma.sync` 内核，在 sm_89 上可用。需要重新转一份
   FP8/INT4 权重。
4. **降模型规模**：8GB 显存放不下 27B/35B。只能用 **3B～8B** 的 Qwen3.x 变体，配合
   INT4/FP8 量化 + 小 KV cache。

GEMM 内核本身是 `mma.sync`（sm_70+）实现，**天然兼容 sm_89**（bf16/f16/s8/tf32 均已实测通过），
这是好消息——不需要重写算子。

---

## 1. 现状诊断（根因）

| # | 根因 | 位置 | 是否阻塞 sm_89 |
|---|------|------|----------------|
| 1 | 构建门禁：CMake 强制 `120a` 且拒绝其它 arch | `CMakeLists.txt:3-13` | ✅ 硬阻塞 |
| 2 | NVFP4 + TMA 内核是 sm_90+/sm_100+ 专属（TMA 在 sm_90+，NVFP4 在 sm_100+） | `src/CMakeLists.txt:57-70`（`ninfer_nvfp4_non_rdc`，TMA 库）、`ops/linear/nvfp4/nvfp4_w4a4_tma.cu`、`ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu` | ✅ 硬阻塞（TMA PTX 在 sm_89 下 nvcc 直接报错） |
| 3 | 发布权重均为 NVFP4（Blackwell 专属格式） | `model-cards/*/artifact-manifest.json: "cuda_architecture":"sm_120a"` | ✅ 需重转（但转换工具支持 FP8/INT4） |
| 4 | 显存：27B/35B 模型权重远超 8GB | `eval/run_qwen3_8_27b_nvfp4_reasoning.sh:20` 注释 | ✅ 需换小模型 |
| 5 | 调优常量 `kRtx5090SmCount=170`、`kRtx5090DramGBs=1792` | 仅在 `bench/ops/*` 与 `eval/corpora/.../data/ninfer/01.txt`（数据文件，不进引擎） | ⚠️ 仅影响 bench 显示与代价模型，不影响正确性 |
| 6 | 代价模型只有 5090 预设 | `src/runtime/engine/context_cache/context_cost_defaults.cpp:51` | ❌ 引擎已有 `generic_*` fallback，未知机型自动走保守估计 |

**关键验证**：`KvCacheStorage` 枚举已含 `BFloat16 / Int8Group64 / Fp8E4M3Row256`（但**无** sm_89 的
NVFP4），见 `include/ninfer/types.h:30-36`。即 KV cache 在 sm_89 上可用 INT8/FP8/BF16。

**架构不阻塞项**：内核用 `mma.sync`（非 `wgmma`），Ada 完全支持。检索确认 `src/` 内**没有**
`wgmma` 指令使用（仅文档/bench 中出现），所以核心 GEMM 路径可原样编译到 sm_89。

---

## 1.5 方案复核（实测验证，2026-09-15）

用本机 CUDA 13.2 的 `ptxas -arch=sm_89` 对每个可疑 PTX 指令逐条实测（PTX 指令的架构支持
在各平台上一致，故 Windows 端 ptxas 的结果对 WSL2 构建同样成立）：

| 指令 / 形式 | sm_89 | sm_90 | sm_100a | sm_120a | 出现位置 |
|-------------|-------|-------|---------|---------|----------|
| `mma.sync ... bf16/f16/s8/tf32` | ✅ | ✅ | ✅ | ✅ | 全量 GEMM |
| `ldmatrix.sync.aligned.m8n8.*` | ✅ | ✅ | ✅ | ✅ | 全量 GEMM |
| `cp.async.cg` + commit/wait | ✅ | ✅ | ✅ | ✅ | 通用流水 |
| `mma ... e4m3.e4m3.f32`（**经典式**） | ✅ | ✅ | ✅ | ✅ | **本方案新增** |
| `mma.sync.aligned.kind::f8f6f4...` | ❌ | ❌ | ✅ | ✅ | `ops/common/mma.cuh:62` |
| `mma.sync.aligned.kind::mxf4nvf4...` | ❌ | ❌ | ✅ | ✅ | `ops/common/mma.cuh:89` |
| `setmaxnreg.*` | ❌ | ✅ | ✅ | ✅ | 仅 NVFP4/TMA 文件 |
| `mbarrier.init / arrive` | ✅ | ✅ | ✅ | ✅ | 仅 NVFP4/TMA 文件 |
| `mbarrier.try_wait.parity` | ❌ | ✅ | ✅ | ✅ | `ops/common/mbarrier.cuh:21` |
| `mbarrier.arrive.expect_tx` | ❌ | ✅ | ✅ | ✅ | `ops/common/mbarrier.cuh:37` |
| `fence.mbarrier_init.release.*` | ❌ | ✅ | ✅ | ✅ | `ops/common/mbarrier.cuh:44` |
| `cp.async.bulk`（TMA） | ❌ | ✅ | ✅ | ✅ | 仅 NVFP4/TMA 文件 |
| `cvt.rn.satfinite.e2m1x2`（fp4） | ❌ | ❌ | ✅ | ✅ | 仅 NVFP4 文件 |

**复核结论（对原方案的修正与补强）**：

1. ❌ **新增阻塞 B3**：`ops/common/mma.cuh:62` 的 FP8 mma 用了 `kind::f8f6f4`（Blackwell 专属）。
   该函数被 **FP8 GEMM**（`ops/linear/fp8/fp8_a8_mma.cuh`）和 **FP8 注意力**
   （`prompt_fp8.cuh`、`small_t_fp8.cuh`）调用——即原方案推荐的 FP8 路线**在当前代码下无法**在
   sm_89 编译。**修复**：`#if __CUDA_ARCH__ >= 1000` 用 `kind::f8f6f4`，否则用经典
   `e4m3.e4m3.f32`（实测经典式在 120a 也通过，双架构均安全；保留 Blackwell 原生式不影响 5090 性能）。
2. ✅ **好消息**：`mbarrier.cuh` 只被 3 个 NVFP4/TMA 文件 include，其余 sm_90+ 原语
   （TMA、setmaxnreg、fp4 cvt）也**只存在于 NVFP4 家族内**——排除 NVFP4 即可一并消除，
   无需为通用注意力内核重写同步。
3. ✅ **INT4 路径安全**：q4/q5/q6/q8 不使用 int4 mma（`s4`），而是解码后走 `mma_s8`（sm_89 通过）。
4. ⚠️ **k8v4 同族**：`small_t_k8v4.*` / `prompt_k8v4.*` / `kv_cache/append/k8v4_launch.cu` 与
   `KvCacheStorage::Fp8KeyNvfp4Value` 依赖 NVFP4 codec，须与 NVFP4 一起排除。
5. ✅ **共享内存**：注意力常量均 < 100KB（如 `kCausalPromptFp8SmemBytes==92416`），
   小于 sm_89 的 99KB/block 上限；`q8_rowsplit_gemm_mma.cuh:58` 已有 `<=99KB` 断言。
6. ✅ `__grid_constant__`（`gdn/.../recurrent.cuh:688`）为 sm_70+，不受影响。

> 复核未发现其他 sm_90+ 构造：`src/` 内无 `wgmma` / `tcgen05` / `elect.sync` / `fence.proxy` /
> `stmatrix` / `cluster` / `mapa` 的使用。

---

## 2. 目标硬件事实（nvidia-smi 实测）

```
name            : NVIDIA GeForce RTX 4060 Laptop GPU
memory.total    : 8188 MiB (~8 GB)
compute_cap     : 8.9   (sm_89, Ada Lovelace)
driver_version  : 596.36
```

对比 RTX 5090：170 SM / 1792 GB/s / 32 GB → 4060：24 SM / ~256 GB/s / 8 GB。
带宽约 1/7，SM 数约 1/7，显存 1/4。**预期推理吞吐约为 5090 的 1/5～1/7**，decode 受带宽限制最明显。

---

## 3. 总体策略

```
RTX 4060 (sm_89)
   ├─ 构建目标: -arch=sm_89        (CUDA 13.1 支持)
   ├─ 排除:     全部 nvfp4/* + TMA 内核
   ├─ 保留:     bf16 / fp8 / q4 / q6 / q8  (mma 内核)
   ├─ 权重:     用 tools/convert 转 fp8_row 或 groupwise(Q4)
   ├─ 模型:     3B / 4B / 8B Qwen3.x（14B int4 勉强，27B/35B 不可行）
   └─ 运行:     WSL2 (Ubuntu 24.04) + 同驱动 596.36，复用 Windows 端 CUDA 驱动
```

平台说明：README 要求 64-bit Linux，且依赖 `pkg-config` 的 ffmpeg/libcurl。原生 Windows 编译
可行但摩擦大，**推荐 WSL2**——CUDA 在 WSL2 下复用 Windows 驱动 596.36，直接装 CUDA 13.1
toolkit + Ninja 即可，几乎零改动。

---

## 4. 分步实施计划

### 步骤 1 — 构建解耦（核心改动）

**1a. 根 CMakeLists 放宽 arch 门禁**
`CMakeLists.txt:3-13`：把单值 `120a` 改为受支持的 arch 列表校验，并导出一个
`NINFER_ENABLE_NVFP4` 谓词（仅当目标含 `100/100a/120/120a` 时为 ON）。

```cmake
# 旧：强制 120a 并 FATAL_ERROR 其它值
# 新：
set(NINFER_SUPPORTED_ARCHS 89 90 100 100a 120 120a)
if(NOT CMAKE_CUDA_ARCHITECTURES IN_LIST NINFER_SUPPORTED_ARCHS)
  message(FATAL_ERROR "NInfer supports ${NINFER_SUPPORTED_ARCHS}; got '${CMAKE_CUDA_ARCHITECTURES}'")
endif()
# NVFP4 仅在 Blackwell 目标可用
set(NINFER_ENABLE_NVFP4 OFF)
foreach(arch 100 100a 120 120a)
  if(arch IN_LIST CMAKE_CUDA_ARCHITECTURES)  # 简化：实际需按前缀匹配
    set(NINFER_ENABLE_NVFP4 ON)
  endif()
endforeach()
```

**1b. `src/CMakeLists.txt` 条件包含 NVFP4 源集**
用生成器表达式把第 61-70 行的 `ninfer_nvfp4_non_rdc` 库，以及后续所有 `nvfp4_*` 文件
（共约 50 个，行号清单见 §6）包进 `$<BOOL:${NINFER_ENABLE_NVFP4}>` 条件。
`src/CMakeLists.txt:359` 的 `target_link_libraries(... ninfer_nvfp4_non_rdc)` 同样加条件。

**1c. 守卫 NVFP4 调用点（避免链接期 undefined reference）**
排除源集后，下列调用点必须在非 Blackwell 构建下不可达：

| 文件 | 行 | 调用 |
|------|----|------|
| `src/ops/linear/linear.cpp` | 6, 96, 140 | `#include nvfp4_dispatch.h`、`nvfp4_dispatch(...)`、`nvfp4_linear_workspace_capacity_bytes(...)` |
| `src/ops/wrapper/linear_swiglu.cpp` | 118 | `validate_nvfp4_weight(...)` |
| `src/ops/wrapper/attn_input_proj.cpp` | 120 | `validate_nvfp4_weight(...)` |
| `src/ops/wrapper/linear_add.cpp` | 215 | `validate_nvfp4_weight(...)` |
| `src/ops/wrapper/gdn_input_proj.cpp` | 305, 427, 573 | `validate_nvfp4_weight(...)` ×3 |
| `src/ops/softmax_attention/dense/causal_cache/small_t.cu` | 238, 388, 430 | `case Nvfp4Group16:` 分支 |
| `src/ops/softmax_attention/dense/causal_cache/prompt.cu` | 71, 98 | `if (cache.storage == Nvfp4Group16)` |
| `src/ops/kv_cache/append/kv_cache_append.cpp` | 203 | `else if (cache.storage == Nvfp4Group16)` |
| `src/ops/kv_cache/append/launch.cu` | 194 | `if (cache.storage == Nvfp4Group16)` |

做法：用 `#ifdef NINFER_ENABLE_NVFP4` 包裹这些调用/分支（枚举值 `Nvfp4Group16` 本身保留，
仅不进入该分支即可）。`switch` 的 `case` 标签若对应函数被排除，需改为 `if/else` 守卫或
`#ifdef` 包裹。

### 步骤 2 — 权重重转（FP8 / INT4）

转换工具确认支持两类数值算法（见 `tools/convert/quantization/__init__.py`）：
- `fp8_row.py` → **FP8 E4M3 row-scaled**（sm_89 有 FP8 tensor core ✅）
- `groupwise.py` → **Q4/Q5/Q6/Q8**（INT4 最适合 8GB 带宽受限场景 ✅）

转一份 8B（或 3B/4B）模型的 INT4 权重：

```bash
python -m tools.convert \
  --model Qwen3.6-8B  \
  --method groupwise --group 64 --bits 4 \
  --out out/qwen3_6_8b.q4060_w4g64.qus
# 或 FP8：
python -m tools.convert --model Qwen3.6-8B --method fp8_row --out out/qwen3_6_8b.q4060_fp8.qus
```

> 注意：文件名里的 `q5090` 只是历史命名，`.qus` 格式与 arch 无关，格式写在 artifact 内部，
> 加载时按格式选内核。新转的权重不会被 `sm_120a` 限制。

### 步骤 3 — 显存预算（8GB 机型）

| 模型 | 参数量 | BF16 | INT8 | **INT4** | FP8 | sm_89 可行性 |
|------|--------|------|------|----------|-----|--------------|
| Qwen3.x-3B  | ~3B  | 6.0 GB | 3.0 GB | **1.5 GB** | 3.0 GB | ✅ 宽松 |
| Qwen3.x-4B  | ~4B  | 8.0 GB | 4.0 GB | **2.0 GB** | 4.0 GB | ✅ 推荐 |
| Qwen3.x-8B  | ~8B  | 16 GB ✗ | 8.0 GB | **4.0 GB** | 8.0 GB | ✅ INT4/INT8 可行（留 ~3-4GB 给 KV+激活） |
| Qwen3.x-14B | ~14B | 28 GB ✗ | 14 GB ✗ | **7.0 GB** | 14 GB ✗ | ⚠️ INT4 勉强，KV 空间极小 |
| Qwen3.6-27B | 27B  | ✗ | ✗ | 13.5 GB ✗ | ✗ | ❌ 放不下 |
| Qwen3.6-35B-A3B (MoE) | 35B | ✗ | ✗ | 17.5 GB ✗ | ✗ | ❌ 放不下 |

**建议默认组合**：8B + INT4（group 64）≈ 4GB 权重，剩余 ~3.5GB 给 KV cache（约 4-8k tokens）
与激活。KV 用 `Int8Group64` 或 `Fp8E4M3Row256` 进一步省显存。

运行参数示例：
```bash
ninfer --artifact out/qwen3_6_8b.q4060_w4g64.qus \
       --kv-cache int8 --max-context 4096 --max-concurrency 1
```

### 步骤 4 — 代价模型（可选优化）

`context_cost_defaults.cpp` 已对未知机型回退到 `generic_context_*`（保守但可用），不影响正确性。
可选：新增一个 `nvidia-geforce-rtx-4060-sm89` 预设（用 `tools/hbm_bandwidth_probe.cu`
实测 RTX 4060 的 DRAM 带宽填入 `ns_per_byte`），让调度更优。非阻塞。

### 步骤 5 — 验证

1. 配置：`cmake -S . -B /build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=89 ...`
2. 编译：确认 nvfp4/TMA 文件被跳过，无 TMA PTX 报错。
3. 加载：用步骤 2 的 INT4/FP8 权重启动，确认 `LoadSummary.weight_formats` 不含 nvfp4。
4. 推理：`ninfer` 跑一段 prompt，确认输出正常、无数值崩溃。
5. bench（可选）：`bench/ops/*` 的 roofline 常量仅显示用，按 §6 把 `kRtx5090DramGBs` 等
   调成 4060 实测值以便看百分比。

---

## 5. 性能预期（诚实估计）

| 维度 | RTX 5090 | RTX 4060 (sm_89) | 倍数 |
|------|----------|-------------------|------|
| SM 数 | 170 | 24 | ~1/7 |
| 显存带宽 | 1792 GB/s | ~256 GB/s | ~1/7 |
| 显存 | 32 GB | 8 GB | 1/4 |
| 8B INT4 decode 吞吐 | 高 | 约 1/5～1/7 | — |

- decode（带宽受限）受 ~1/7 带宽拖累最重；prefill（compute 受限）受 SM 数拖累。
- INT4 比 FP8 更省带宽，在 8GB 上**优先 INT4**。
- 不要期待接近 5090 的体验，但 8B 模型跑通日常对话/推理完全可行。

---

## 6. 需改动/守卫的文件清单（实现阶段用）

> 本节为**方案期的预估清单**；实际落地后的准确清单与守卫位置见 **§9.1**（已修订）。

**构建系统**
- `CMakeLists.txt`（根，3-13 行）
- `src/CMakeLists.txt`（57-70, 112, 116, 126, 146-172, 203-210, 240-243, 262-265, 359）

**NVFP4 调用点守卫**（见 §4 步骤 1c 表格）

**转换**
- `tools/convert`（仅使用，无需改；`quantization/groupwise.py`、`fp8_row.py` 已就绪）

**可选（代价模型预设）**
- `src/runtime/engine/context_cache/context_cost_defaults.cpp:51`（新增 4060 预设）
- `tools/hbm_bandwidth_probe.cu`（改 `-arch=sm_89` 测 4060 真实带宽）

---

## 7. 风险与缓解

| 风险 | 说明 | 缓解 |
|------|------|------|
| `__launch_bounds__` / 寄存器文件差异 | sm_120 寄存器文件 128K，sm_89 仅 64K；个别 mma 内核可能寄存器不足报错 | 编译后若出现 "too many resources"，下调对应内核 launch_bounds 或 split 参数 |
| 内核假设 170 SM 做 occupancy 计算 | 部分 prefill 持久化 block 数按 5090 算 | 影响性能非正确性；必要时按 24 SM 重新计算 |
| CUDA 13.1 在 WSL2 装包 | 需匹配驱动 596.36（已满足） | 用官方 runfile / apt 源，别混版本 |
| 原生 Windows 编译 | 项目依赖 pkg-config ffmpeg/libcurl，Windows 下需 vcpkg 或手动 | 推荐 WSL2，避免此坑 |
| 8B INT4 KV 溢出 | 长上下文下 KV 占满 8GB | 默认 `--max-context 4096`，KV 用 int8/fp8 |

---

## 8. 验证清单（含落地后的实测结果）

| 项 | 结果 | 证据 |
|----|------|------|
| `-DCMAKE_CUDA_ARCHITECTURES=89` 不再 FATAL_ERROR | ✅ 已验证 | `cmake -P` 断言：89→NVFP4=OFF，120a→NVFP4=ON，90/100/75/121 被拒 |
| NVFP4 内核不出现在 sm_89 目标 | ✅ 已验证 | `src/CMakeLists.txt` 生成器表达式门控；§10.4 预处理实测 |
| 无 `undefined reference to nvfp4_*` | ✅ 静态+预处理证明 | §10.4：m=0 时 9 个文件对排除内核的**调用点为 0** |
| FP8 路径在 sm_89 可编译 | ✅ 已验证 | `mma.cuh` 架构守卫实测：`__CUDA_ARCH__=890` 只出经典式 |
| 守卫代码在两种宏值下均能编译 | ✅ 已验证 | g++ 16.1 `-fsyntax-only` × 7 文件 × 2 宏值 = 14/14 通过 |
| KV 存储 FP4 在非 Blackwell 被拒 | ✅ 已验证 | `paged_kv_storage.h` 预处理：m=0 含拒绝分支，m=1 不含 |
| 端到端全量链接 | ⚠️ 本机不可执行 | 见 §10.5：本机缺 Windows SDK，且 `cmd.exe`/`wsl.exe` 被安全策略禁用 |
| 转换出 INT4/FP8 权重并跑通推理 | ⏳ 待你在可构建环境执行 | §4 步骤 2 / §10.6 |

---

## 9. 落地实现记录（2026-09-15）

### 9.1 已完成的改动

**构建系统**

| 文件 | 改动 |
|------|------|
| `CMakeLists.txt`（根） | ① 支持列表收敛为 `89;120a`（两个已实测验证的目标），非法 arch 的报错信息明确要求"验证后再扩列表"；② `NINFER_ENABLE_NVFP4` 谓词（`100/100a/120/120a` → ON）；③ 改为**项目级** `add_compile_definitions(NINFER_ENABLE_NVFP4=0/1)`，消除"哪个 target 拿到了宏"的隐患 |
| `src/CMakeLists.txt` | ① 新增 `NINFER_NVFP4_GUARD` 生成器表达式；② `ninfer_nvfp4_non_rdc` 整个 target 包在 `if(NINFER_ENABLE_NVFP4)` 内；③ 全部 NVFP4/k8v4 源文件用 `$<${NINFER_NVFP4_GUARD}:...>` 门控；④ 移除原先的 per-target `target_compile_definitions`（已上移到根） |
| `build-sm89.bat`（新增） | 一步完成 sm_89 的 configure/build；`VSDEVCMD`、`CUDA_PATH` 可用环境变量覆盖。**注意**：需已安装 VS + Windows SDK |

**源码守卫（共 11 个文件）**

| 文件 | 守卫位置 |
|------|----------|
| `src/ops/common/mma.cuh` | `mma_fp8_e4m3`：`__CUDA_ARCH__>=1000` 用 `kind::f8f6f4`，否则经典式 |
| `src/core/paged_kv_storage.h` | **单一收口点**：非 Blackwell 构建拒绝 `Nvfp4Group16` / `Fp8KeyNvfp4Value` |
| `src/ops/linear/linear.cpp` | dispatch + workspace 容量，共 2 处 |
| `src/ops/wrapper/linear_swiglu.cpp` | 头文件 + dispatch + workspace 容量，共 3 处 |
| `src/ops/wrapper/linear_add.cpp` | 头文件 + dispatch + workspace 容量，共 3 处 |
| `src/ops/wrapper/attn_input_proj.cpp` | 头文件 + dispatch + workspace 容量，共 3 处 |
| `src/ops/wrapper/gdn_input_proj.cpp` | 头文件 + 3 个 dispatch（proj/snapshot/record）+ 3 个 workspace 容量，共 8 处 |
| `src/ops/kv_cache/append/kv_cache_append.cpp` | k8v4/nvfp4 存储分派 |
| `src/ops/kv_cache/append/launch.cu` | 批量 append 的 k8v4/nvfp4 分派 |
| `src/ops/softmax_attention/dense/causal_cache/prompt.cu` | 2 处分派 |
| `src/ops/softmax_attention/dense/causal_cache/small_t.cu` | 4 处分派 |
| `src/serve/serve_options.cpp` | `parse_kv_dtype` 早失败：非 Blackwell 直接拒绝 `--kv-dtype nvfp4\|k8v4` |

### 9.2 三个关键设计决策

1. **FP8 用"架构守卫"而非"换形式"**。实测经典 `e4m3.e4m3.f32` 在 sm_89/90/100a/120a **全部通过**，
   但直接全局替换会改变 5090 的 codegen。用 `#if __CUDA_ARCH__ >= 1000` 双分支：5090 保持原有
   `kind::f8f6f4` 路径位级不变，Ada 走经典式——**零回归风险**。

2. **k8v4 与 NVFP4 共用同一守卫**。实测 `cvt.rn.satfinite.e2m1x2.f32` 在 sm_89 **和 sm_90 都不可用**，
   要到 **sm_100a** 才有——所以 k8v4 与 NVFP4 同为 Blackwell 专属，用同一个 `NINFER_ENABLE_NVFP4`
   门控是语义正确的，不是"勉强合并"。

3. **"单一收口点 + 分支编译剔除 + 选项层早失败"三层防御**，而非在每个 `.cu` 里写异常：
   - `paged_kv_storage_layout()` 是**所有** KV-cache 路径（分配 / 规划 / 校验 / 启动）的必经点，
     在此拒绝 FP4 存储 → 覆盖完整且报错明确；
   - `.cu` 内的分派分支直接 `#if` 剔除（`.cu` 里抛异常会牵入 `<stdexcept>` 依赖，
     且这些分支本就不可能到达）；
   - `serve_options.cpp` 在解析 `--kv-dtype` 时就失败，用户不会等到加载模型才看到错误。

   这与 `AGENTS.md` 的既有约定一致：*"Native preparation, resource queries and execution enforce
   actual Op support; there is no whole-artifact capability registry."* —— 即在 Op/执行层
   强制能力，而非在 artifact 层建能力注册表。因此 `artifact/formats.cpp`、`artifact/materializer.cpp`、
   `core/weight.h`、`core/weight_view.*` 中出现的 `QType::NVFP4`（纯枚举与几何校验，与架构无关）
   **有意不加守卫**。

### 9.3 未改动（有意保留）

- `KvCacheStorage` 枚举本身（`include/ninfer/types.h:35`）——公开 API，保留枚举值，
  由运行时拒绝，而非破坏 ABI。
- `serve/operational_log.cpp`、`serve/request_log.cpp`、`causal_softmax_attention.cpp:361-366`、
  `small_t.cu:238` 中的 FP4 引用——仅枚举→字符串映射或启发式常量，不影响编译或正确性。

---

## 10. 验证证据（实测）

### 10.1 PTX 指令 / 架构矩阵
见 §1.5 表格（本机 `ptxas` 13.2 逐一实测）。

### 10.2 CMake 门禁逻辑
`cmake -P` 断言 8/8 通过：`89→OFF`、`120a→ON`、`90/100/75/121` 被拒、`89/120a` 被接受。

### 10.3 守卫源码双向编译（真实编译器）
`g++ 16.1 -std=c++20 -fsyntax-only`，对 7 个宿主文件分别以 `NINFER_ENABLE_NVFP4=0` 与 `=1` 编译：

```
NINFER_ENABLE_NVFP4=0 : linear.cpp / linear_swiglu.cpp / linear_add.cpp /
                        attn_input_proj.cpp / gdn_input_proj.cpp /
                        kv_cache_append.cpp / serve_options.cpp   全部 PASS
NINFER_ENABLE_NVFP4=1 : 同上 7/7 全部 PASS
```

> 这一步实际抓出并修复了 3 处 `#if` 未闭合的真实缺陷（`gdn_input_proj.cpp` 的
> conv_snapshot / record 两处 dispatch 与 record workspace 容量），说明该检查有效而非走过场。

### 10.4 守卫有效性（预处理级）
- `mma.cuh` 分支选择：`__CUDA_ARCH__=890` → 只出经典 `e4m3`；`=1000`/`=1200` → 出 `kind::f8f6f4`。✅
- `paged_kv_storage.h`：`m=0` 输出含架构拒绝分支，`m=1` 不含。✅
- **排除内核调用点为 0**（对预处理产物做调用点分析，剔除声明）：

  | 文件 | m=0 调用点 | m=1 调用点 |
  |------|-----------|-----------|
  | `prompt.cu` | **0** | 4 |
  | `small_t.cu` | **0** | 4 |
  | `kv_cache/append/launch.cu` | **0** | 2 |
  | `kv_cache_append.cpp` | **0** | 2 |
  | `linear_swiglu.cpp` | **0** | 2 |
  | `linear_add.cpp` | **0** | 2 |
  | `attn_input_proj.cpp` | **0** | 2 |
  | `gdn_input_proj.cpp` | **0** | 5 |

  → sm_89 构建不会引用任何被排除的内核符号，**不会出现 `undefined reference`**。

### 10.5 本机全量构建的真实阻塞（已更正）

**更正**：本文档早期版本写「本机缺 Windows SDK」——**这是错的**。当时只查了默认路径
`C:\Program Files (x86)\Windows Kits\10`（不存在），而实际 SDK 装在 **D 盘**：

```
注册表 HKLM\SOFTWARE\WOW6432Node\Microsoft\Microsoft SDKs\Windows\v10.0
  InstallationFolder = D:\Windows Kits\10\      (ProductVersion 10.0.26100)
  → D:\Windows Kits\10\Include\10.0.26100.0\um\Windows.h  ✅
```

实际就位情况：`cl.exe`（MSVC 14.44，含 `msvcrt.lib`）、`vcvars64.bat`、`nvcc` 13.2、
`cmake` 3.28+、`ninja` —— **全部可用**。

**真正的阻塞是宿主第三方依赖**：根 `CMakeLists.txt:85-87` **无条件**要求

```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
```

即使 `-DNINFER_BUILD_APPS=OFF` 也绕不开。本机 `pkg-config` 存在
（`F:\AI\w64devkit\bin\pkg-config`），但 `libavformat/libavcodec/libavutil/libswscale/libcurl`
**全部 MISS**，且无任何 `libav*` 开发库 → **CMake 配置必然失败**。
需要 vcpkg/MSYS2 安装 ffmpeg + libcurl 开发库，或改用 WSL2。

另：`cmd.exe` / `wsl.exe` 在本会话被安全策略禁用（Program Blacklist），故无法走 `vcvars64.bat`；
但 **PowerShell 可用**，可手工拼装 MSVC/SDK 的 `INCLUDE`/`LIB`/`PATH` 后直接调用 `nvcc`/`cmake`。

### 10.6 真实设备编译（新增，比预处理检查强得多）

即使 CMake 配置因缺 ffmpeg 失败，**仍可对每个 `.cu` 做真实 nvcc 设备编译**（产出 cubin）。
这能抓出预处理级检查看不到的 PTX/ISA/模板错误。本机实测可用：

```bash
NVCC="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/nvcc.exe"
CCBIN="D:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
# 必须导出 INCLUDE(MSVC include + SDK ucrt/um/shared/winrt) 与 LIB
"$NVCC" --use-local-env -ccbin "$CCBIN" -cubin -arch=sm_89 -std=c++20 -lineinfo \
  -DNINFER_ENABLE_NVFP4=0 -Iinclude -Isrc -Ithird_party -Ithird_party/utf8proc \
  -Xcompiler /wd4819 src/path/file.cu -o out.cubin
```

两个必须加的开关（踩坑记录）：
- **`--use-local-env`**：否则 nvcc 会尝试运行 `vcvars64.bat` 来建立 VS 环境，而 `cmd.exe` 被禁用
  → `nvcc fatal : Could not set up the environment for Microsoft Visual Studio using '.../vcvars64.bat'`。
- **`-ccbin <MSVC bin>`**：仅把该目录加进 bash 的 `PATH` 不够，nvcc 仍报 `'cl.exe' 不是内部或外部命令`。

### 10.7 在可构建环境里跑通（WSL2 / 补齐 ffmpeg 开发库）

```bash
cmake -S . -B build-sm89 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=OFF
cmake --build build-sm89 --parallel
# 期望配置阶段输出：NInfer CUDA architectures: 89 (NVFP4 kernels: OFF)
```
Windows + 已装 SDK 且补齐 ffmpeg/libcurl 开发库后，可直接用 `build-sm89.bat`。

---

## 11. 第二轮复核：改变计划的新发现（2026-09-15 晚）

复核目标：构建解耦已完成，验证路径上还剩哪些**硬约束**。结论有两面——原方案第 3 条
「降模型规模」**低估了成本**；但「加载器可能还卡架构」这一担忧**被排除**。

### 11.1 ❌ 形状表是硬约束（原方案漏项）

六个权重格式的 linear 分派都在**枚举表**里查 `(n, k)`，查不到直接抛异常，**没有通用兜底**：

| 文件 | 行为 |
|---|---|
| `src/ops/linear/q4/q4_dispatch.cpp:13-32` | 10 条 `ShapeEntry`，否则 `throw "q4 linear: unsupported shape"` |
| `src/ops/linear/q5/q5_shapes.h` | 7 个 `select_q5_nY_kZ` |
| `src/ops/linear/q6/`、`bf16/` | 同结构（另有 `shapes/` 实例化目录） |
| `src/ops/linear/q8/q8_dispatch.cpp:35-40` | 同结构 |
| `src/ops/linear/fp8/fp8_dispatch.cpp:9-16` | `kFp8N14336K5120` 等，否则抛异常 |

而现有表覆盖的几何**只有 hidden=5120 这一族**（q4 表：`1024/4096/5120/6144/7168/34816/131072 × 5120`，
外加 vision 的 `3456/4304 × 1152`）。
→ **换一个 hidden≠5120 的模型，即使权重转好了也跑不起来。** 这是比构建门禁更靠后的阻塞。

### 11.2 ✅ 但扩表很便宜（约 15 行/形状）

形状条目不含 kernel 实现，只是**在已有的通用 launch 变体之间按 token 数挑一个**：

```cpp
// src/ops/linear/q6/shapes/n1152_k1536.cpp —— 全文 15 行
Q6Launch select_q6_n1152_k1536(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0) throw ...;
    if (tokens <= 96)  return launch_q6_simt_r8_c4;
    if (tokens <= 704) return launch_q6_mma_r64_c64;
    return launch_q6_mma_r64_c128;
}
```

底层 kernel 是**几何无关**的磁贴变体（`simt_r8_c4` / `mma_r64_c64` / `mma_r64_c128`）；
`Q8LinearGeometry<N,K>`（`q8_geometry.h`）只要求 `N%16==0 && K%32==0`。
→ 新增一个模型几何 ≈ **加一条表项 + 一个选择器 + 调 token 阈值**，不是重写算子。

**对齐前提**：`K` 需为 128 的倍数（现有表项全部满足：5120 / 6144 / 2048 / 1152 / 1536）。
绝大多数 LLM 的 hidden 满足此条件。

> ⚠️ **本节结论已被 §14.2 修正（2026-09-15 深夜）**：`linear/*/*_dispatch.cpp` 只是
> **其中一处**收口点。`attn_input_proj` / `gdn_input_proj` / `linear_add` / `linear_swiglu` /
> `dynamic_grouped_conv` 等算子各自还有 `supported_shape()` **精确问题**谓词，把
> `input_rows / padded_k / qk_rows / ...` 写死到具体模型，且是 27B 与 35B-A3B **各写一套**。
> 因此"扩表 = 15 行/形状"仅适用于 `linear` 一族，**不适用于整条执行链**。

### 11.3 ✅ 加载器不校验架构（排除一处担忧）

`cuda_architecture` 只出现在 `model-cards/*/artifact-manifest.json`（值 `sm_120a`），
`src/` 内**无任何校验代码**（`grep cuda_architecture src/` 零命中）。
→ 不需要改 artifact 加载器，也不需要重签 manifest。

### 11.4 另两条影响下一步的事实

- **代价模型**：`context_cost_defaults.cpp:48-80` 内置预设只有 `nvidia-geforce-rtx-5090-sm120`；
  未知机型走 `generic_context_transfer_cost()` / `generic_context_prefill_cost()`（注释明确
  "preserve numerical ranking"，不是关闭代价模型）。**已有运行时覆盖通道**
  `--context-cost-presets FILE`（`serve_options.cpp:182`）→ 4060 标定**无需改代码**。
- **官方 recipe / 模型卡只有 27B 与 35B-A3B**（`model-cards/` 5 张卡、`official_recipes.py` 5 个
  recipe），**没有小模型**。8GB 的硬算术：27B 的 q4/q5 混合产物实测 **17,495,537,920 字节
  （约 16.3 GiB）** → 8GB 无解。按 `q4_g64 ≈ 4.25 bit/param` 估算，8GB 内可承载的参数上限
  约 **8B**，且还需留出 KV 与工作区。

---

## 12. 下一步计划（分阶段）

> **一句话**：代码侧的构建解耦已完成；下一步的**唯一关键动作是先做阶段 0.2**——确认是否存在
> 可用的同架构小模型。它决定 8GB 目标是否可达，比"把构建跑起来"更关键。

### 阶段 0 — 现在就能做（不需要构建环境）

| # | 事项 | 产出 / 判据 |
|---|---|---|
| 0.1 | **产品范围决策**：把 sm_89 列为受支持目标，属于 AGENTS.md 意义上的产品变更，需你确认 | 确认后我才改 `README.md` / `AGENTS.md` |
| 0.2 | **确定目标小模型**：需存在同架构（`Qwen3_5ForCausalLM`）、hidden 为 128 倍数、能拿到的 checkpoint | 一个源模型路径，或"此路不通"的明确结论 |
| 0.3 | **形状表扩展清单**：把目标模型的每个 `(n,k)` 映射到具体文件与选择器模板 | `doc/` 下新增扩展清单，可直接照做 |

**若 0.2 结论为「没有可用小模型」**，8GB 目标不可达，替代选项只有：换机器 / 放弃本地运行 /
把交付范围收敛为「sm_89 构建支持已就绪」（即当前状态）。

### 阶段 1 — 构建环境与首次真实构建

| # | 事项 | 完成判据 |
|---|---|---|
| 1.1 | WSL2 装 CUDA 13.1（驱动 596.36 已满足），**或**给原生 Windows 补装 Windows SDK（二选一） | `nvcc --version` 可用 |
| 1.2 | sm_89 构建（§10.6） | 配置行显示 `(NVFP4 kernels: OFF)`；链接**零** `nvfp4_*` 未定义符号 |
| 1.3 | 120a 回归构建 | 默认架构构建通过；NVFP4 路径与改动前一致 |

### 阶段 2 — 权重与首次运行

| # | 事项 | 完成判据 |
|---|---|---|
| 2.1 | 用 `tools/convert` 转 `q4_g64`（推荐）或 `fp8_row` | 产物 manifest 的 `formats` 不含 `nvfp4` |
| 2.2 | 若几何不在形状表中 → 按 §0.3 清单扩表并重建 | 不再出现 `unsupported shape` |
| 2.3 | 8GB 预算核对：权重 + KV（int8/fp8）+ 工作区 | 启动后显存留有安全余量 |
| 2.4 | 冒烟推理 | 输出通顺，无 CUDA 错误 |

启动参数建议：`--kv-dtype int8`（或 `fp8`）、`--max-context 4096` 起步；
**不要**用 `nvfp4` / `k8v4`——sm_89 构建会在早失败点给出明确报错。

### 阶段 3 — 调优（可选，不阻塞可用性）

- 3.1 用 `--context-cost-presets` 提供 4060 标定 JSON（机制已有，无需改代码）
- 3.2 按实际负载调整形状表内的 token 阈值与磁贴选择

### 阶段 4 — 文档收口（必须在验证绿了之后）

- 4.1 `README.md:31-34` 现写「requires … RTX 5090 … **The build rejects CUDA architectures other
  than `sm_120a`**」——**已与代码不符**；`AGENTS.md:32-33` 的架构承诺同样过时。
  按 AGENTS.md 的 change consistency 必须同步更新，并声明 sm_89 的能力边界
  （无 NVFP4 / 无 k8v4 / FP8 走经典 `e4m3.e4m3.f32` 形式）。
  **建议等阶段 1.2 绿了之后再改**，避免文档先于验证。
- 4.2 把本文档 §11/§12 更新为最终状态

---

## 13. 第二轮落地：真实设备编译暴露的阻塞与修复（2026-09-15 晚）

> **重要**：本节推翻了之前"代码侧已基本完成"的乐观判断。用本机 nvcc 做**真实设备编译**
> （§10.6 的方法）后，发现 **21 个文件在 sm_89 下编不过**，其中 **11 个是真实阻塞**。
> 之前只做预处理级检查是看不到这些的。

### 13.1 方法与首轮结果

对 `src/` 下 **153** 个（`m=0` 应编译的）`.cu` 逐个 `nvcc -cubin -arch=sm_89`：

```
=== RESULT sm_89 macro=0: PASS=132 FAIL=21 / 153 ===
```

### 13.2 三类失败（已逐一分类，非偶发）

把 21 个失败**串行**重跑（排除并行干扰）并按错误内容分类：

| 类别 | 数量 | 真实原因 | 性质 |
|---|---|---|---|
| `C1189: MSVC/cl.exe with traditional preprocessor` | 10 | MSVC 传统预处理器下 CCCL/CUB 拒绝编译 | **Windows 宿主标志问题**，与 sm_89 无关；加 `-Xcompiler /Zc:preprocessor` 后**全部转 PASS** |
| `griddepcontrol ... requires .target sm_90 or higher` | 4 | PDL（programmatic dependent launch） | ✅ **已修复**（§13.3） |
| `uses too much shared data (…, 0xc000 max)` | 7 | sm_89 的**静态**共享内存上限 **48 KB**（`0xc000`） | ⚠️ **未修复**，需设计决策（§13.4） |

受 PDL 影响的文件：
`ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu`、`…_independent.cu`、
`ops/sparse_moe/decode/sparse_moe_decode_kernels.cu`、`ops/sparse_moe/small_t/sparse_moe_small_t_kernels.cu`；
另有 4 个共享头（`q4/q5_rowsplit_gemv.cuh`、`q4/q5_rowsplit_gemm_simt.cuh`）也引用 PDL。

受静态共享内存影响的 7 个文件**全部是 q8**：
`attn_input_proj/q8/q8_attn_input_gemm_splitk.cu`、`dynamic_grouped_conv/q8/…add_materialized.cu`、
`linear/q8/shapes/n2048_k4096.cu`、`linear/q8/shapes/n2048_k16384.cu`、
`linear_pair/q8/q8_pair_gemm_splitk.cu`、`linear_swiglu/q8/q8_linear_swiglu_gemm_mma.cu`、
`linear_add/q8/q8_linear_add_gemm_splitk.cu`。

### 13.3 ✅ 已修复：PDL / `griddepcontrol`（sm_90+ 专属）

**为什么之前漏掉**：上一轮的 ISA 扫描只查了 `mbarrier` / `cp.async.bulk` / `stmatrix` / `elect.sync` /
`cluster` 等**字面量**，而 PDL 是通过内建函数 `cudaGridDependencySynchronize()` /
`cudaTriggerProgrammaticLaunchCompletion()` 引入的，源码里搜不到 `griddepcontrol` 字样。
`core/pdl.cuh` 是**唯一收口点**，因此修复只改了这一个头文件 + 一个 CMake 谓词：

- 根 `CMakeLists.txt`：新增 `NINFER_ENABLE_PDL` 谓词（`90/90a/100/100a/120/120a` → `1`，否则 `0`），
  经 `add_compile_definitions` 项目级下发；配置行现在会打印 `(NVFP4 kernels: …; PDL: …)`。
- `src/core/pdl.cuh`：
  - 设备侧 `trigger_dependents()` / `wait_for_dependencies()` 在非 sm_90+ 时**编译为空操作**
    （用 `__CUDA_ARCH__ >= 900` 判定，宿主 pass 回退到 `NINFER_ENABLE_PDL`）。
  - 宿主侧 `launch_dependent()` 在不支持时**不下发** `programmaticStreamSerialization` 属性
    → 恢复普通流序语义。

**语义安全性**：PDL 只是把生产者尾部与消费者头部做重叠的**调度优化**；去掉它意味着消费者必须在
生产者完成后才启动——**只损失重叠，不改变任何数值结果**。因此这里"静默降级"是正确的
（与 NVFP4 必须**响亮失败**不同：那是权重格式选择，静默替换会改变数值）。

**验证**：4 个原失败文件在 sm_89 下**全部转为 PASS**；随后全量重跑 →
`PASS=146 FAIL=7 / 153`（21 → 7，其中 10 个靠 `/Zc:preprocessor`、4 个靠 PDL 修复）。

### 13.4 ⚠️ 未修复：q8 内核静态共享内存超过 sm_89 的 48 KB

**报错形态**（同一内核的多个模板实例）：
```
ptxas error : Entry function '…q8_ksplit_grouped_mma_kernel<…>' uses too much
              shared data (0x11000 bytes, 0xc000 max)
```
`0xc000 = 49152 B = 48 KB` —— sm_89 的**每 block 静态共享内存上限**。
请求量实测在 `0xc200`(49.7KB) ~ `0x16000`(88KB) 之间。

**根因**：这些内核用**静态** `__shared__` 数组，而项目按 sm_120a 调优，静态容量可达 ~99KB：
- `ops/linear/q8/q8_ksplit_grouped_mma.cuh:34-35`：
  `__shared__ std::uint8_t code_shared[kMmaRows][kGroupK]` + `b_shared[kKernelWarps][kWarpCols*kTileK]`
- `ops/linear/q8/q8_rowsplit_gemm_mma.cuh:58,98`：
  `static_assert(SMEM_BYTES <= 99*1024, "sm_120a per-CTA shared memory limit")` + `__shared__ SharedStorage shared`
- `ops/linear/q8/q8_ksplit_mma.cuh:108-110`：已同时具备 `extern __shared__` 动态缓冲（可复用其模式）

**可选方案**（需你选一个；这是本阶段唯一剩下的架构性工作量）：

| 方案 | 做法 | 代价 |
|---|---|---|
| **A（推荐）** | 把静态 `__shared__` 改为 `extern __shared__` 动态共享内存，并在启动时设置 `cudaFuncAttributeMaxDynamicSharedMemorySize`；sm_89 经 opt-in 可用到 99 KB | 需改 ~7 个内核/启动器；最大请求 88KB < 99KB 可行，但 sm_89 每 SM 仅 100KB → 占用率降为 1 block/SM，性能下降 |
| **B** | 保留静态，为 sm_89 增加一组把 `kTileK`/阶段数/`KSplits` 压到 ≤48KB 的实例化配置 | 机制较简单，但需新增按架构区分的配置表，且吞吐损失可能比 A 更大 |
| **C** | sm_89 下剔除 q8 的这些变体，让 q8 走其余路径 | 最省事，但 q8 权重（27B 卡里确有 `q8_g32_fp16`）在 4060 上会失效或大幅降速 |

> 说明：在 4060 上推荐主用 **q4/q5**（表内 q4 路径已全部编译通过），q8 主要用于少数敏感权重。
> 因此方案 A/B 的必要性取决于你是否打算在 4060 上用 q8 权重。

### 13.5 Windows 宿主的额外一项（与架构无关）

`C1189` 那 10 个文件说明：**用 MSVC 做原生 Windows 构建时**，需要给 cl 加标准符合预处理器开关，
否则 CCCL/CUB 拒绝编译。CMake 侧一行即可（仅 MSVC 生效）：

```cmake
if(MSVC)
  add_compile_options(/Zc:preprocessor)
endif()
```

在 WSL2/Linux 构建时**不需要**（gcc/clang 默认符合标准）。我没有擅自加入——它属于 Windows
宿主可移植性，不是本次 sm_89 适配的必需项，等你确认构建平台后再定。

### 13.6 修正后的当前状态

| 项目 | 状态 |
|---|---|
| 构建门禁（arch 列表 + `NINFER_ENABLE_NVFP4` 谓词） | ✅ 完成 |
| NVFP4 / k8v4 剥离与三层能力防御 | ✅ 完成 |
| FP8 `mma_fp8_e4m3` 经典形式回退 | ✅ 完成 |
| **PDL / griddepcontrol sm_90+ 阻塞** | ✅ **本轮发现并修复**，sm_89 实测通过 |
| **q8 静态共享内存 48KB 上限** | ⚠️ **未修复**，7 个文件，需选方案 A/B/C |
| sm_89 设备编译 | **146 / 153 PASS**（剩余 7 个即上述 q8） |
| sm_120a 回归 | 见 §13.7 |
| 全量链接 + 推理 | 仍需宿主第三方开发库（ffmpeg/libcurl）或 WSL2 |

### 13.7 sm_120a 回归：183 / 183 全通过

同一套设备编译方法跑 5090 目标（`macro=1`、`PDL=1`、全部 183 个 `.cu` 含 NVFP4/TMA 内核）：

```
=== RESULT sm_120a macro=1: PASS=183 FAIL=0 / 183 ===
```

两点结论：

1. **5090 路径未被本次改动影响**：NVFP4 / TMA / k8v4 全部内核照常编译，PDL 保持开启
   （`NINFER_ENABLE_PDL=1`），`mma.cuh` 的 `kind::f8f6f4` 分支在 sm_100+ 下仍走原生形式。
2. **反证 §13.4 的诊断**：那 7 个 q8 文件在 sm_120a 下全部通过 → 失败确系 sm_89 的
   **48 KB 静态共享内存上限**，而非代码本身有错。

### 13.8 验证强度对照（本轮提升）

| 层次 | 上一轮 | 本轮 |
|---|---|---|
| ISA 级（ptxas 单指令探测） | ✅ | ✅（补充 `griddepcontrol` 一项） |
| 构建逻辑级（cmake 断言） | ✅ | ✅ |
| 宿主 `.cpp` 编译（g++，双宏值） | ✅ 14/14 | ✅ |
| 预处理级（调用点归零） | ✅ | ✅ |
| **真实设备编译（nvcc -cubin）** | ❌ 未做 | ✅ **sm_89 146/153、sm_120a 183/183** |
| 全量链接 + 端到端推理 | ❌ | ❌ 仍需 ffmpeg/libcurl 开发库或 WSL2 |

**教训**：预处理级检查（上一轮的"调用点归零"）**不足以**证明一个架构移植可用——它看不到
ISA 指令、静态共享内存上限这类**编译器后端**约束。必须做真实设备编译。

---

## 14. 第三轮复核（2026-09-15 深夜）：小模型问题已解决，但适配面比预估大得多

> 本轮目标：把 §12 阶段 0 的两个未知（0.2 目标小模型、0.3 形状表缺口）落定，并复测构建环境。
> 结果：**0.2 得到肯定答案（8GB 目标变为可达）**，但 **§11.2 的乐观估计被推翻**。

### 14.1 ✅ 阶段 0.2 已解决：存在可用小模型 `Qwen3.5-9B`

官方 Qwen3.5 **Small 系列**（2026-03-01/02 发布）：`0.8B / 2B / 4B / 9B`，Apache 2.0，
262K 原生上下文，与 27B/35B-A3B 同属 **Qwen3.5 混合 GDN + Attention** 架构——
即 NInfer 实现的 `Qwen3_5ForCausalLM`。**Qwen3.6 / Qwen3.8 家族则没有小尺寸**
（只有 27B Dense、35B-A3B、以及 122B/397B/2.4T 级别），所以目标必须落在 3.5 Small 上。

**`Qwen/Qwen3.5-9B` 几何**（HF 模型卡 + transformers 参考配置）：

| 项 | 值 | 与现有表的关系 |
|---|---|---|
| `hidden_size` | 4096 | ❌ **不在任何表内**（现有 `k ∈ {5120,6144,17408,2048,1152,1536,4304}`） |
| `intermediate_size` | 12288 | ❌ 不在表内（27B 是 17408） |
| `vocab_size` | 248320 | LM head 需 `n248320 × k4096`（现有 `n248320 × k5120/2048`） |
| `num_hidden_layers` | 32（`full_attention_interval=4` → 24 GDN + 8 full） | — |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 16 / 4 / 256（RoPE 64） | 新组合，需确认 attention 核是否按 head 几何参数化 |
| GDN：`linear_num_key_heads` / `linear_num_value_heads` / dim / conv | 16 / 32 / 128 / 4 | ✅ **与 27B、35B-A3B 完全一致** |
| Vision encoder（`depth`/`hidden`/`intermediate`） | 27 / 1152 / 4304 | ✅ 已被 q4/q5/q6 表覆盖（跨模型共享） |
| MTP | 有（1 层） | ✅ 与现有 spec 路径同构 |

**8GB 预算**：按 `q4_g64 ≈ 4.25 bit/param`，9B ≈ **4.8 GB** 权重（q5 约 5.7 GB），
剩余约 2.5–3 GB 给 KV（`int8`/`fp8`）+ 工作区 + CUDA 上下文 → **8GB（实测 8188 MiB）可行**。

> 之所以之前判定"8GB 无解"，是因为当时只看了 `model-cards/` 与 `official_recipes.py`——
> 那里确实只有 27B/35B-A3B 的五张卡；小模型存在于**上游 HF**，缺的是 NInfer 侧的 recipe 与准入几何。

### 14.2 ⚠️ 修正 §11.2：形状适配面远大于"6 张 linear 表"

逐算子复查后确认：`linear/*/*_dispatch.cpp` 只是**其中一处**。每个「算子 × 量化格式」还有自己的
`supported_shape()` **精确问题（exact problem）谓词**，把张量行数写死到具体模型：

| 算子 / 格式 | 位置 | 写死的精确问题 |
|---|---|---|
| `attn_input_proj` q4_q5 | `q4_q5_attn_input_plan.cpp:10-13` | `input_rows==5120 && query_rows==6144 && kv_rows==1024 && padded_k==5120` |
| `gdn_input_proj` q4_q5 | `q4_q5_gdn_input_plan.cpp:41-44` | `input_rows==5120 && qk_rows==4096 && value_z_rows==12288 && qkv_rows==10240 && z_rows==6144 && padded_k==5120` |
| `attn_input_proj` q8 | `q8_attn_input_plan.cpp:77-82` | `input_rows==2048 && padded_k==2048 && query 4096/kv 512/parent 9216`，另有 companion/dflash2 分支 |
| `gdn_input_proj` q8 | `q8_gdn_input_plan.cpp:38-41` | `input_rows==2048 && qkv_rows==8192 && z_rows==4096 && parent_rows==12288 && padded_k==2048` |
| `linear_add` q5 | `q5_linear_add_plan.cpp:72-80,119` | `kSupports` 表 + `k∈{6144,17408}` 两套路由分支 |
| `linear_add` q8 | `q8_linear_add_plan.cpp:216,235` | 同结构 |
| `linear_swiglu` q8 | `q8_linear_swiglu_plan.cpp:134,154` | 同结构 |
| `dynamic_grouped_conv` q8 | `q8_dynamic_grouped_conv_add_plan.cpp:15` | `C` 必须为 `4096` 或 `17408` |
| `linear` q4/q5/q6/q8/bf16/fp8 | `*_dispatch.cpp`（§11.1 已列） | `(n,k)` 枚举 + `throw "unsupported shape"` |

**决定性观察**：上表前两行是 `…==5120`（27B），后两行是 `…==2048`（35B-A3B）——
说明适配是"**每模型 × 每格式 × 每算子**各写一份准入"，而不是共享一张几何无关的表。
§11.2 里"扩表 ≈ 15 行/形状"只对 `linear` 一族成立，**对整条执行链不成立**。

**对 9B 的实际工作量**（仍需落地验证，此处为核对后的估计）：

- 需新增准入条目：约 **8 个算子 × 2 个主用格式（q4_q5 / q8）≈ 16 处**，外加 `linear` 的
  `(n,k)` 表项（`k=4096` 一族：`n ∈ {1024(kv),4096(q/gate),12288(mlp up/gate),248320(lm head)}`；
  `k=12288, n=4096`（mlp down））。
- 每个新准入通常还需对应的 **kernel 模板实例化 + 选择器（token 阈值）**。
- ✅ **不需要重写内核数学**：GDN 头几何与 vision 几何跨模型一致，数学部分不变。
- ⚠️ q8 一侧还要先解决 §13.4 的 48 KB 静态共享内存问题；**但 8GB 装不下 q8 权重**
  （9B q8 ≈ 9 GB > 8188 MiB，27B q8 更不可能），因此 9B 目标可以先只做 **q4/q5**，
  q8 在 sm_89 上直接走方案 C（剔除）。

### 14.3 构建环境复测（2026-09-15）

| 项 | 状态 |
|---|---|
| `nvcc` / CUDA | ✅ 13.2.78（`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`） |
| GPU | ✅ RTX 4060 Laptop GPU，**8188 MiB**，driver 596.36，`compute_cap 8.9` |
| `cmake` / `ninja` | ✅ `C:\Program Files\CMake\bin\cmake.exe`、`C:\Python312\Scripts\ninja.exe` |
| MSVC / Windows SDK | ✅ MSVC 14.44 + SDK `D:\Windows Kits\10`（10.0.26100） |
| `pkg-config` | ⚠️ 仅有 w64devkit 自带 0.34（`F:\AI\w64devkit\bin\pkg-config.exe`） |
| **ffmpeg 开发库**（libavformat/codec/util/swscale） | ❌ **全缺**，全盘无 `libavformat.pc` |
| **libcurl 开发库** | ❌ **缺**，无 `libcurl.pc` |
| vcpkg / MSYS2 | ❌ 均未安装 |
| Docker | ⚠️ 已装但**守护进程未运行**（`dockerDesktopLinuxEngine` 管道不存在）；Linux 容器依赖 WSL2 |
| WSL2 | ❌ 不可用（`wsl.exe` 被安全策略禁用） |

→ 本机唯一可执行的构建路径是 **原生 Windows + MSVC**，且必须先取得 **MSVC 版**
ffmpeg + libcurl 开发库（vcpkg，或 gyan.dev 的 ffmpeg shared 构建 + curl-for-win，
后者还需自备 `.pc` 文件或改 CMake 用 `find_package`）。或者由你自行放开 WSL2。

### 14.4 剩余待办清单（本轮更新）

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| 0.1 | 产品范围决策：把 sm_89 列为受支持目标 | ⬜ 待你确认 | 确认后才改 `README.md`/`AGENTS.md` |
| 0.2 | 确定目标小模型 | ✅ **已解决 → `Qwen3.5-9B`** | 见 §14.1 |
| 0.3 | 准入表 / 形状表扩展清单（逐算子枚举到文件级） | 🔶 骨架已完成（§14.2） | 需补：9B 每个投影的确切 `(n,k)` 与算子准入元组 |
| A | q8 静态共享内存超 48 KB（7 文件，§13.4） | ✅ **已解决（选 C）** | sm_89 下剔除这 7 个 q8 内核；q4/q5 不经过它们 |
| B | **ffmpeg + libcurl 的 MSVC 开发库**（或 WSL2） | ✅ **已解决** | gyan.dev ffmpeg shared + libcurl 8.22 源码自建 + 手写 `.pc` |
| C | `if(MSVC) add_compile_options(/Zc:preprocessor)` | ✅ **本轮已加入根 CMakeLists** | 原生 Windows 构建必需；WSL2/Linux 无影响 |
| D | `Qwen3.5-9B` 的 recipe（官方或自定义） | ⬜ 未做 | 与 0.3 并行 |
| E | 首次真实全量构建（配置行应打印 `NVFP4 kernels: OFF; PDL: OFF`） | ✅ **已完成**（见 §15.1） | rc=0，三个 exe 产出且可启动 |
| F | 首次推理 + 8GB 预算核对（建议 `--kv-dtype int8 --max-context 4096`） | ⬜ 未做 | 依赖 D、0.3 —— **硬约束见 §15.6** |
| G | 4060 代价模型预设 JSON | ⬜ 可选 | 机制已有（`--context-cost-presets`），不改代码 |
| H | `README.md:31-34` / `AGENTS.md:32-33` 架构声明同步 | ⬜ 未做 | 建议等 E 绿了之后 |
| I | 本次 15 个已改文件提交 | ⬜ 未做 | — |

### 14.5 建议执行顺序

1. **先解 B**（ffmpeg + libcurl 的 MSVC 开发库）——唯一"不解决就彻底卡住"的项。
2. 跑通 **E**：确认 sm_89 全量构建除 q8 外全绿（把 §13 的"设备编译 146/153"升级为真实全量构建结论）。
3. 再定 **A**：若只在 4060 上跑 9B 的 q4/q5，直接选 **C**，在 sm_89 下剔除那 7 个 q8 内核。
4. 然后做 **D + 0.3**（9B recipe + 准入表扩展）→ **F** 冒烟推理。
5. 最后 **H / I** 收口（文档与提交）。

**数据来源**：`Qwen/Qwen3.5-9B` HF 模型卡与参考配置（`hidden_size=4096`、
`intermediate_size=12288`、`num_key_value_heads=4`、`head_dim=256`、GDN 16/32 头 ×128、
`full_attention_interval=4`）；NVIDIA Megatron-Bridge Qwen3.5 文档（确认
`Qwen/Qwen3.5-0.8B/2B/4B/9B` 仓库存在）；Qwen3.6 / Qwen3.8 家族构成。

---

## 15. 第四轮：sm_89 全量构建通过并首次运行成功（2026-09-15 深夜）

### 15.1 结果

`D:\deps\build\ninfer-89` 全量构建 **rc=0**，产出
`apps/ninfer.exe` / `apps/ninfer-serve.exe` / `apps/ninfer-perplexity.exe`，
**且能真正启动运行** —— §13 的"设备编译 146/153"结论已升级为**真实全量构建 + 运行**结论。

配置阶段输出（与预期一致）：

```
-- NInfer CUDA architectures: 89 (NVFP4 kernels: OFF; PDL: OFF; large static __shared__: OFF)
```

### 15.2 交付汇总：本轮解决的问题

| 现象 | 原代码 / 根因 | 修复 |
|---|---|---|
| 8 个 TU 报 `error C4003`/`C2589`/`C2059`（`min`/`max` 宏） | NVTX 的 `nvtxImpl.h` 在 `_WIN32` 下**无条件** `#include <windows.h>`；链路 `core/weight.h → nvtx.h → nvToolsExt.h → nvtxImpl.h` | 根 `CMakeLists.txt` 的 `if(MSVC)` 块加**项目级** `add_compile_definitions(NOMINMAX)` |
| `logging.cpp` / `perplexity/main.cpp` 报 `localtime_r`/`gmtime_r` 找不到标识符 | 直接调 POSIX 时间函数 | 新增 `platform::local_calendar_time` / `utc_calendar_time`（MSVC 用 `localtime_s`/`gmtime_s`，**参数顺序相反**） |
| `materialization_planner.h` 报 `__uint128_t` 未声明 | 用 `unsigned __int128` 做 64×64 精确乘积比较 | 新增 `core/saturating.h` 的 `wide_multiply` / `wide_compare`（高低 32 位拆分） |
| `ninfer.exe` 链接 `LNK2019`：`getaddrinfo`/`freeaddrinfo`/`inet_ntop`/`ntohl` | `acquire.cpp` 不经 cpp-httplib，缺 Winsock 链接 | `src/CMakeLists.txt` 在 `if(WIN32)` 下给 `ninfer_media_acquire` 加 `ws2_32` |
| **构建全绿但一给模型路径就 `0xC0000005` 崩溃**（崩在 `MSVCP140.dll`，日志/初始化阶段，无任何输出） | 见 §15.3 —— **不是代码问题** | 部署匹配版本的 VC143 CRT（`msvcbuild.py deploycrt 89`） |

### 15.3 根因：本机 System32 的 VC 运行时是错版 + 混版（会复发，务必记住）

| DLL | System32 实测 | 需要 |
|---|---|---|
| `msvcp140.dll` | **14.00.24215.1**（VS2015 U3，文件时间 2026-07-12） | ≥ 14.40 |
| `vcruntime140.dll` | **14.00.24215.1** | ≥ 14.40 |
| `msvcp140_1/_2/atomic_wait/codecvt_ids`、`vcruntime140_1/_threads` | 14.50.35719.0 | — |

而编译用的是 `14.44.35207` 的头文件。**MSVC 14.40 起 `std::mutex` 与
`std::condition_variable` 的默认构造变成 `constexpr`**（`include/mutex:35-46` 的
`_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` 两支），就地存储改由**运行时惰性初始化**，
不再调用 `_Mtx_init_in_situ`。14.40 之前的 `_Mtx_do_lock` 仍按旧约定办事，
解引用仍为 NULL 的 `_Critical_section._M_srw_lock`（即 `std::mutex` + 8）→ AV。

**决定性证据（也是排除误判的关键）**：`sizeof(std::mutex)=80`，偏移 0 = `_Type` = `2`
（= `_Flags|_Mtx_try`，与 `__msvc_threads_core.hpp:43-49` 的布局完全吻合），偏移 8 = 0。
**修复前后对象字节完全相同** —— 对象没坏，坏的是"运行时的期望"。
（若只看崩溃位置，极易误判成自己的构造函数有 bug。）

修法：把工具集自带的 `Microsoft.VC143.CRT` 复制到 exe 旁（**应用程序目录优先于 System32**），
已固化为 `msvcbuild.py deploycrt 89`。
注意 Redist 目录版本号与工具集目录**不同**（`14.44.35112` vs `14.44.35207`），要 glob 取最新。

> ⚠️ **每次重建构建树 / clean 之后都要重跑一次 `deploycrt`**，否则崩溃会"复现"。
> 根治需管理员权限（未做）：把 System32 那两个 DLL 换成 ≥ 14.40 的版本。

> ❌ 不要选 `-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR`：旧 DLL **确实导出**
> `_Mtx_init_in_situ` / `_Cnd_init_in_situ`（`dumpbin /exports` 已确认）→ 能链接，
> 但那等于把数值引擎跑在 2016 年的 CRT 上，且需全量重建（宏必须全 TU 一致）。

### 15.4 运行验证证据

```
$ msvcbuild.py env apps\ninfer.exe <dummy>.bin --prompt hi
starting engine
error: startup failed | starting engine | 1.7s
error: NInfer accepts only .ninfer artifacts       ← 干净的应用级拒绝
rc=1                                                ← 不再是 0xC0000005

$ msvcbuild.py env apps\ninfer.exe --help
rc=0
```

`<dummy>.bin` 是 21 字节假文件，故意触发"非 .ninfer"分支，用来证明已越过 Engine 构造。

### 15.5 待办清单（本轮更新）

| # | 事项 | 状态 |
|---|---|---|
| A | q8 静态共享内存超 48 KB | ✅ **已解决**（选 C：sm_89 下剔除 7 个 q8 内核） |
| B | ffmpeg + libcurl 的 MSVC 开发库 | ✅ **已解决**（gyan.dev ffmpeg shared + libcurl 8.22 源码自建 + 手写 `.pc`） |
| C | MSVC 编译开关（`/Zc:preprocessor`、`/utf-8`、`NOMINMAX`） | ✅ **已完成** |
| E | 首次真实全量构建 | ✅ **已完成**（rc=0，三个 exe） |
| — | 构建产物可启动（运行验证） | ✅ **已完成**（§15.4） |
| — | 临时插桩清理（`ck`/`lck`/crash filter/`/MAP`） | ✅ **已完成**（重建并复跑验证） |
| 0.1 | 产品范围决策：把 sm_89 列为受支持目标 | ⬜ 待你确认 |
| 0.3 | 9B 的准入表 / 形状表扩展（逐算子到文件级） | ⬜ 未开始（骨架见 §14.2） |
| D | `Qwen3.5-9B` 的 recipe | ⬜ 未开始（官方 recipe 只有 27B / 35B-A3B，需自定义） |
| F | 首次推理 + 8GB 预算核对 | ⬜ 依赖 D + 0.3 |
| G | 4060 代价模型预设 JSON | ⬜ 可选（机制已有，不改代码） |
| H | `README.md` / `AGENTS.md` 架构声明同步 | ⬜ 未做 |
| I | 本次已改文件提交 | ⬜ 未做 |

### 15.6 关于 F 的一个硬约束（动手前必须先知道）

`src/ops/linear/linear.cpp` 证明**没有任何通用兜底**：所有量化格式
（`q4`/`q5`/`q6`/`q8`/`bf16`/`nvfp4`/`fp8`）都进入各自的 `select_*_launch()` 闭表，
而且 `linear_workspace_capacity_bytes()` 在**规划期**就调用同一批选择器。

q4/q5 表内全是 27B 几何（`k=5120`）与 vision 几何（`k=1152`）
→ 9B（`hidden_size=4096`）会在**规划阶段**就
`throw std::invalid_argument("q4 linear: unsupported shape")`。

所以 **F 不是"下载权重 + 转换"就能跑通**：必须先完成 0.3
（每个投影的新 `(n,k)` 入表 + token 阈值选择器），再谈转换与冒烟推理。

> ⚠️ 本段初版把 0.3 描述为"kernel 实例化 + token 阈值选择器"，**高估了工作量**。
> 读实现后修正，见 §16.2。

---

## 16. 第九轮：sm_89 构建的"能跑到什么程度"验证（2026-09-16）

前一轮解决了"构建全绿但一运行就崩"（本机 System32 VC 运行时错版，§15.3）。本轮回答
一个更实际的问题：**现在到底能跑什么？**

### 16.1 结论：内核层面能跑，模型层面还不能

| 层次 | 状态 | 证据 |
|---|---|---|
| 全量 CMake 构建（`sm_89`） | ✅ 能 | rc=0，三个 exe（§15.2） |
| 二进制启动 / 参数解析 | ✅ 能 | `--help` rc=0；带路径不再崩（§15.4） |
| **CUDA 内核在真机执行** | ✅ **能** | **6/6 测试在 RTX 4060 上 Passed**（见 16.3） |
| **加载并推理一个模型** | ❌ **不能** | 无 `.ninfer` 产物 + 可用模型几何不在闭表内（见 16.4） |

### 16.2 修正：0.3 不是"内核实例化"，而是"加表行"

§15.6 初版判断过重。读实现后的事实：

- `Q4Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t)`
  —— **n、k 不在内核里，运行时从 `Weight` 读**。
- `src/ops/linear/q4/shapes/*.cpp` 是**纯 token 数 → tile 配置的选择表**，且**复用通用 launcher**
  （`launch_q4_gemv_r1_q8_direct` / `launch_q4_simt_r8_c4` / `launch_q4_simt_r8_c8` /
  `launch_q4_mma_r64_c128`）。`n4096_k5120.cpp` **全文只有 4 行 `if`，无任何新内核**。
- 只有少数 `.cu` 形状（`n1024_k5120` / `n5120_k6144` / `n6144_k5120` / `n131072_*`）会
  `launch_q4_ksplit<N,K,T>` / `launch_q4_mma<Schedule>` 把 n、k 烤进模板 —— 那些是**调优**，
  **不是可运行性的前提**。

→ **最小可跑路径** = 给新几何加"指向通用 launcher 的选择器 + 表行"：每形状约 5 行 `.cpp`
+ `q4_shapes.h` 一行声明 + `q4_dispatch.cpp` 一行表项。专用 tile 的调优可以后补。
成本量级从"大规模内核实例化"降到"批量加表行 + 准入谓词"。

### 16.3 在 RTX 4060 上的实测（新增验证手段）

`BUILD_TESTING=ON` 后在**现有构建树**上增量编译（复用已编译对象，不重编库）：

```
$ msvcbuild.py configure 89 -DBUILD_TESTING=ON
-- NInfer CUDA architectures: 89 (NVFP4 kernels: OFF; PDL: OFF; large static __shared__: OFF)
$ msvcbuild.py targets 89 ninfer_device_test ninfer_linear_q4_a16_test \
      ninfer_linear_q5_a16_test ninfer_linear_swiglu_q4_a16_test \
      ninfer_linear_add_q4_a16_test ninfer_linear_add_q5_a16_test      → rc=0
$ msvcbuild.py deploycrt 89
$ msvcbuild.py env ctest --test-dir D:/deps/build/ninfer-89 -R "..." --output-on-failure

ninfer_device_test .................   Passed    0.25 sec
ninfer_linear_q4_a16_test ..........   Passed    8.24 sec
ninfer_linear_q5_a16_test ..........   Passed    4.69 sec
ninfer_linear_add_q4_a16_test ......   Passed    3.30 sec
ninfer_linear_add_q5_a16_test ......   Passed    2.41 sec
ninfer_linear_swiglu_q4_a16_test ...   Passed    3.79 sec

100% tests passed, 0 tests failed out of 6          Total Test time = 22.69 sec
```

**为什么这是强证据**：这些 op 测试用 `tests/ops/quantized_weight.h` **确定性生成** packed 权重，
并用 `tests/ops/op_tester.h` 的**独立数学 oracle** 校验 —— **不需要任何真实模型**。
所以它直接证明了 `sm_89` 编出的 q4/q5 矩阵乘、`linear_add`、`linear_swiglu` 内核
**在 Ada 硬件上数值正确**，而不只是"能链接"。

⚠️ `deploycrt` 已扩展为同时覆盖 `apps/` 与 `tests/`（测试 exe 同样链接动态 VC 运行时，
缺匹配 CRT 会在第一个 `std::mutex` 加锁时崩，见 §15.3）。**重建构建树后必须重跑。**

### 16.4 运行期阻塞的完整清单（为什么还不能跑模型）

1. **磁盘上没有任何 `.ninfer` 产物**。HF 缓存里 `models--turboderp--Qwen3.5-9B-exl3` 与
   `models--UnstableLlama--Qwen3.6-35B-A3B-exl3-4.00bpw` 各只有 1 KB，是**空占位目录**。
2. **27B / 35B-A3B 在 8 GB 上不可行**。官方 `tools/convert/official_recipes.py` 只有 5 条 recipe
   （27B ×3、27B-nvfp4 ×2、35B-A3B ×1），权重分别 ≈15 GB / ≈20 GB。且全仓库**没有权重 offload**
   ——`host_resident` 只出现在 KV-cache 页的放置逻辑（`program/planning/pressure.cpp`、
   `checkpoint_recovery.cpp`），**模型权重必须全部驻留显存**。
3. **唯一可行的 9B（`hidden_size=4096`）不在任何闭表内** → 规划期即
   `throw std::invalid_argument("q4 linear: unsupported shape")`。

### 16.5 9B 投影几何（从 `tools/convert/qwen3_5.py` 推导，非猜测）

来源：`attention()` / `gdn()` / `dense()` 的宽度公式，而非文档推断。

| 组件 | 公式 | 9B 取值（`h=4096`） |
|---|---|---|
| `attention/query` | `(num_attention_heads * head_dim, h)` | `(4096, 4096)` |
| `attention/gate` | 同 query（源 `q_proj` 为 `(2q,h)` 拆半） | `(4096, 4096)` |
| `attention/key`、`/value` | `(num_key_value_heads * head_dim, h)` | `(1024, 4096)` |
| `attention/output` | `(h, q)` | `(4096, 4096)` |
| GDN `qkv` | `channels = 2*kg + vg`，`kg=16*128`、`vg=32*128` | `(8192, 4096)`（跨模型不变量） |
| GDN `z` / `output` | `vg` / `h` | `(4096, 4096)` |
| `mlp/gate` + `mlp/up` | **成组融合**（27B 表内为 `(34816,5120)` = 2×17408） | `(24576, 4096)`；非融合 `(12288, 4096)` |
| `mlp/down` | `(h, intermediate)` | `(4096, 12288)` |
| `token_embedding` / `output_head` | `(vocab, h)` | `(248320, 4096)` |
| vision | hidden 1152 / intermediate 4304 / out 3584 | **与 27B 一致 → 已在表内，无需新增** |

对照实测闭表：`q4` 表 `(n,k)` = `(1024,5120) (4096,5120) (5120,6144) (6144,5120) (7168,5120)
(34816,5120) (131072,5120) (131072,2048) (3456,1152) (4304,1152)`；
`q5` 表 = `(1024,5120) (6144,5120) (7168,5120) (5120,6144) (5120,17408) (1152,1152) (1152,4304)`。
→ **9B 需要的 k=4096 / 12288 与 n=248320 / 24576 全部缺失。**

### 16.6 待办清单（本轮更新）

| # | 事项 | 状态 |
|---|---|---|
| 0.1 | 产品范围决策：把 sm_89 列为受支持目标 | ⬜ 待确认 |
| 0.3 | 9B 的准入表 / 形状表扩展 | ⬜ 未开始（**成本已下调，见 §16.2**；几何见 §16.5） |
| D | `Qwen3.5-9B` 的 recipe | ⬜ 未开始（`qwen3_5.py` 通用，缺的是 recipe 条目 + 下载约 5 GB） |
| F | 首次推理 + 8GB 预算核对 | ⬜ 依赖 0.3 + D |
| G | 4060 代价模型预设 JSON | ⬜ 可选 |
| H | `README.md` / `AGENTS.md` 架构声明同步 | ⬜ 未做 |
| I | 本次已改文件提交（当前 **45** 个） | ⬜ 未做 |
| — | **sm_89 内核真机数值正确性** | ✅ **已验证**（§16.3，6/6 Passed） |
| — | `deploycrt` 覆盖 `tests/` | ✅ 已完成 |
