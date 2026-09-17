# RTX 4060 适配 Qwen3.8-27B：权重部分 CPU 驻留（offload）方案

> 状态：**方案评审中**（未实施）
> 关联：`rtx4060-sm89-9b-adaptation-plan.md`（9B 移植已完成 L1–L3 + attention）
> 日期：2026-09-16

---

## 1. 目标与背景

在 RTX 4060 Laptop GPU（8GB VRAM，sm_89）上运行 Qwen3.8-27B（Dense），显存放不下的权重
驻留主机内存（pinned），按层流送 GPU 计算。

**一个必须先对齐的概念**：NInfer 全部算子均为 CUDA kernel，**没有 CPU 计算路径**。
「部分放到 CPU 上」在本引擎的可行语义是 **权重驻留 CPU 内存 + 数据仍由 GPU 计算**
（每 token 把该层权重经 PCIe 拷入固定显存槽位后执行 kernel），而非把部分层放到 CPU 上算。
后者等于另写一套 CPU 推理引擎，不属于本方案。

## 2. 关键事实

### 2.1 模型几何（官方 config 实测）

| 参数 | 值 | 备注 |
|---|---|---|
| hidden / intermediate | 5120 / 17408 | 与 Qwen3.6-27B 相同 → **op 层闭表已覆盖** |
| 层数 | **64** | 48 GDN + 16 full-attn（每 4 层 1 个 full） |
| full-attn | 24q / 4kv / head_dim 256 | 与 27B causal 几何一致 |
| GDN | key 16头×128、value 48头×128 | value_rows=6144 ✓ |
| vocab | 248320 | 与 3.5 系一致 |
| MTP | 1 层（mtp_num_hidden_layers） | 4060 不启用（Q8/MTP 已剔除） |

### 2.2 体积估算（官方 recipe `qwen3_8_27b`：Q4/Q5 投影 + Q6 词表）

| 部分 | 估算 |
|---|---|
| 每层平均（~383M 参数，Q4/Q5 混合 ≈0.6 B/参数） | **~230 MB × 64 层 ≈ 14.7 GB** |
| embedding + output head（Q6，248320×5120×2） | ~2.1 GB |
| **权重总计** | **≈ 16.8 GB** |

### 2.3 显存预算（4060 Laptop 8GB）

| 项 | 预算 |
|---|---|
| 桌面/驱动占用 | ~0.6 GB |
| CUDA context + workspace + 激活 | ~1.2 GB |
| KV cache（8 个 full 层 32KB/token + GDN 状态，8K ctx） | ~0.5 GB |
| 流送槽位（双缓冲 = 2 层份） | ~0.5 GB |
| **可用于常驻权重** | **≈ 5.2 GB（约 30%）** |
| 需流送部分 | ≈ 11.6 GB/token |

### 2.4 性能预估（诚实数字）

4060 Laptop = PCIe 4.0 **x8**，实测 pinned H2D ≈ 12–14 GB/s。
每 token 流送 11.6 GB → ~0.9 s → **decode ≈ 1.0–1.3 tok/s**。

- 这是**权重带宽瓶颈**，与 kernel 效率基本无关；speculative/MTP 无法改善（瓶颈在搬运不在计算）。
- 若目标是「可日常使用的 27B」，llama.cpp GGUF（本机 `LM_MODEL/lmstudio-community/Qwen3.6-27B-GGUF`）
  的 GPU 层拆分是现成路径（约 3–5 tok/s）。**本方案的价值是引擎能力建设**（offload 基础设施），
  预期管理请以 ~1 tok/s 为准。

## 3. 方案选型

| 方案 | 说明 | 结论 |
|---|---|---|
| A. Host 驻留 + GPU 流送（本方案） | 权重大头 pinned 在主存，双缓冲槽位 H2D 与计算重叠 | ✅ 推荐，改造面可控 |
| B. 局部层 CPU 计算 | 需全套 CPU kernel（GEMV/attention/GDN/silu） | ❌ 等于新引擎，不做 |
| C. 外部引擎（llama.cpp GGUF） | 现成 `--n-gpu-layers` | 🔶 要「能用」选它；与本引擎无关 |

## 4. 架构设计（方案 A）

### 4.1 驻留划分

| 权重 | 驻留 | 理由 |
|---|---|---|
| embedding + output head | GPU 常驻 | 每 token 必用，且 Q6 值得保 |
| 前 N 层（按预算装满 ~5.2GB） | GPU 常驻 | 零流送成本 |
| 其余层全部投影权重 | **Host pinned**（16GB+ 主存） | 逐层流送 |

### 4.2 槽位与流水线（双缓冲）

```
transfer_stream:  [H2D 层 i+1 ≈230MB]──────▶[H2D 层 i+2]───▶
compute_stream:   [层 i kernels]──▶[层 i+1 kernels]────────▶
                  （CudaCompletionEvent 做跨流依赖）
```

- 设备端开 **2 个层槽位**（`slots[2]`，每槽 ~230MB，位于 DeviceArena 内），
  交替使用：计算层 i（槽 s）的同时，层 i+1 从 pinned host 拷入槽 s^1。
- 每层的 `Weight.qdata` 改指**槽位地址**（指针恒定、内容轮换）——kernel 零改动。
- 关键约束：`execution/parameters.h:130-133` 注明权重地址 startup 冻结 → 槽位地址
  确实冻结（内容轮换不违反该假设）✓。

### 4.3 CUDA Graph

decode 默认走 graph capture（`program/decode.cpp:24-79`）。offload 模式：
- **P0 直接关闭**：`EngineOptions.use_cuda_graph=false`（`program/planning/startup.h:82`，开关现成）。
- P2 进阶（可选）：整条 decode 捕获为含 `memcpy` 节点的图——每层固定 host 源地址
  （全部 offload 权重常驻 pinned，每层一个固定地址）+ 槽位复用依赖边，可做到一次
  `graphLaunch` 完成整 token 且自动重叠。风险较高，放最后。

### 4.4 Prefill 差异

prefill 逐 chunk 执行同一条层循环，复用同一套槽位机制；长 prompt 时流送成本被
分摊（每层权重只拷一次/每 chunk），受影响相对小。

## 5. 改造面（探索结论，4 必改 + 复用清单）

### 必改

| # | 位置 | 改动 |
|---|---|---|
| 1 | `src/models/qwen3_5/load/prepare.cpp:25` | 按 `EngineOptions` 的显存预算，把超出部分 object 的 `binder.parameter(..., Residency::Device→Host, ...)`（Binder/materializer **双路 placement 已支持，零新机制**） |
| 2 | `execution/parameters.h` + `execution/text.cpp`（`run_layers:1077` / `attn_mix:840` / `gdn_mix:928` / `mlp_tail:1069`） | 权重指针间接化：进层前若该层为 Host 驻留 → 等待/发起槽位 H2D，`Weight` 视图指向槽位 |
| 3 | 新增 `WeightStreamManager`（建议 `src/runtime/engine/`） | 组装现成件：`PinnedHostBuffer`（`arena.h:93`）+ `transfer_stream`（`device.cu:62`）+ `CudaCompletionEvent`（`device.h:68`）；host 源可直接 `Reader::read_direct`（`artifact/reader.h:44`）文件直读，免 host 全量常驻 |
| 4 | `program/planning/startup.h:82` | offload 模式强制 `use_cuda_graph=false`（P0） |

### 现成可复用（不新建）

- `MaterializedArtifact` 每 object 的 device/host 双槽（`materializer.h:73-77`）
- `bind_view` 按 `Residency` 选 parent（`views.cpp:14-16`）
- device 容量按 placement 自动重算（`artifact/binder.cpp:157-178`）→ Arena 与
  `resolve_kv_capacity`（`runtime/engine/kv_capacity.cpp:79`）**自动**联动缩小
- MoE `next_weight_prefetch` hint 骨架（`sparse_moe.h:35-38`）→ 语义平移为 H2D prefetch
- PDL 在 sm_89 编译为空（`pdl.cuh`，安全降级）→ 重叠只靠双 stream + event

## 6. 分阶段实施

| 阶段 | 内容 | 验收 |
|---|---|---|
| **P0 功能打通** | 改动 #1/#4 + 最简顺序流送（同步 H2D→计算，无重叠）+ `--offload-ratio` CLI | 27B Q4 在 4060 出 token；显存 ≤8GB；数值与常驻模式一致（同一 prompt 贪心解码 diff=0） |
| **P1 双缓冲重叠** | 改动 #2/#3（槽位轮换 + event 依赖） | decode 提升 ≥1.6×（重叠效率 ≥80%） |
| **P2（可选）** | graph + memcpy 节点；常驻层自动选择策略（按层频度/尺寸） | 单 graph replay；预算自动分配 |

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| 槽位复用 vs 「地址冻结」假设的隐性依赖（如图 capture、缓存指针） | P0 关 graph 后全量回归 12 套件；对 offload 模式加专项测试（同权重双路径 diff） |
| 主存压力：offload 16GB pinned + 系统 | 需 32GB RAM；pinned 分配失败要回退可分页 + 警告 |
| PCIe x8 实际带宽不达 12GB/s | P0 先实测带宽探针（`probe.cu` 复用），据实修正预期 |
| KV 容量反推用 `cudaMemGetInfo` 剩余显存 | placement 缩小后自动成立；P0 验证 8K ctx 不 OOM |
| 与 9B 冒烟（Task F）的优先级冲突 | 建议先完成 9B 冒烟（代码已就绪、下载可续传），再开本方案 |

## 8. 验证方案

1. **单测**：新增 `tests/ops/` offload 专项——同一权重 Device 常驻 vs Host 流送，
   输出逐位一致（槽位拷贝路径不引入数值差异）。
2. **带宽探针**：pinned H2D 实测（x8 拓扑确认）。
3. **真机冒烟**：27B Q4 转换件 + `--offload`；指标 = tok/s、峰值显存（`nvidia-smi`）、
   长 prompt prefill 时间；与 §2.4 预估对照。
4. **回归**：12 套件 11/12 基线不破坏（offload 关闭时路径零变化）。

## 9. 前置依赖

- 27B 转换件：需官方 Qwen3.8-27B safetensors（~54GB bf16，走 hf-mirror；本机仅有
  NVFP4-freetoken 与 GGUF，均不可用于本转换器）——**转换本身就是 ~1–2 小时级任务**。
- 或者：用 P0 前先拿 **Qwen3.5-9B 冒烟**（`D:\models\Qwen3.5-9B` 已有 14.3GB 分片，可续传），
  offload 基础设施（P0/P1）先在 9B 上以「强制 30% 常驻」模式验证，再上 27B——**推荐顺序**。
