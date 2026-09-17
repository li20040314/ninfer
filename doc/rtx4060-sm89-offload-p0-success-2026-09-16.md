# P0 offload 功能打通：9B 强制 30% 常驻验证通过

> 状态：**P0 完成**（2026-09-16 晚）
> 关联：`rtx4060-qwen38-27b-cpu-offload-plan.md`（方案）、`rtx4060-sm89-smoke-success-2026-09-16.md`（9B 冒烟）
> 验证物：Qwen3.5-9B artifact `--offload-ratio 0.7`（≈30% 权重常驻、70% 主存流送）

---

## 1. 验收结果（4060 Laptop 8GB）

| 指标 | 基线（全常驻） | offload 0.7 | 结论 |
|---|---|---|---|
| 输出（greedy，同 prompt） | 我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型，具备强大的逻辑推理、代码生成及多模态理解能力。 | **逐字一致（diff=0）** | ✅ 数值零差异 |
| GPU 权重 | 5.28 GiB | **4.21 GiB** | ✅ 驻留缩减 |
| decode | 35.9 tok/s | 5.1 tok/s | 符合预期（PCIe + 同步流送） |
| CUDA Graph allowance | 12.0 MiB | **0 B** | ✅ offload 强制关 graph |
| KV capacity auto | 2048 tok | 2048 tok | ✅ 不受影响 |

流送比例实测：5.28−4.21 = 1.07 GiB/token 流送，decode 5.1 tok/s ⇒ 有效流送带宽 ≈5.5 GB/s
（同步 H2D、页缓存源、无重叠）。P1 双缓冲 + pinned 源预计可再提 ≈1.6×+。

## 2. 实现（与方案 §5 的差异）

方案改动 #1 的实现点从 `prepare.cpp`（Bindings::parameter）后移到 **`binder.finish()` 之后改写
MaterializationPlan**——所有 bind_* 零改动，消除"绑定层不知道层号"的问题：

| # | 文件 | 改动 |
|---|---|---|
| 1 | `src/models/qwen3_5/load.cpp` | `plan_weight_offload()`：按 `weight_offload_ratio` 计算流送层边界（流送字节 ≥ ratio×文本层总字节）；把流送权重的 `ParameterReference.residency` 改为 Host；重写 plan——被移除对象转 `HostPlacement`（materializer 自动从 reader 读入主存），剩余 device 对象**重新排布 offset**（移除中间对象会位移后续分配）；冻结每对象槽位偏移（256B 对齐），产出 `WeightOffloadSpec`（含 per-layer / per-WeightId 槽位映射） |
| 2 | `src/models/qwen3_5/offload_stream.h/.cpp`（新） | `WeightStreamScheduler`：持 `DeviceBuffer` 槽位（一次性 cudaMalloc，尺寸=最大流送层字节）+ 每对象 `WeightParent`（geometry 复制自 host parent，data 指向槽位内偏移）+ `ensure_layer(layer)`：整层对象 `cudaMemcpyAsync`(H2D, transfer_stream) + `cudaStreamSynchronize`（P0 同步语义） |
| 3 | `src/models/qwen3_5/model.h/.cpp` | Model 持有 scheduler（声明序保证先于 backing_ 析构），`offload_stream()` 访问器 |
| 4 | `src/models/qwen3_5/execution/text.{h,cpp}` | `TextContext` 构造时经 `parameters_.model.offload_stream()` 自动接线（6 处构造点零改动）；`run_layers` 循环每层前 `ensure_layer`，prefill/decode/verify 全路径覆盖 |
| 5 | `include/ninfer/types.h` + `src/models/load_options.h` | `EngineOptions.weight_offload_ratio` → `LoadOptions`（适配器补字段） |
| 6 | `src/runtime/engine/model_instance.cpp` | `validate_options` 校验 [0,1)、仅 Generation、拒绝 speculative；`normalize_engine_options` 强制 `use_cuda_graph=false` |
| 7 | `apps/cli/options.{h,cpp}` + `main.cpp` | CLI `--offload-ratio F`；serve 侧 `--offload-ratio` 同步加入 |

关键机制：流送权重的 kernel 可见指针 = **槽位地址**（startup 冻结，内容轮换），故
`prepare_*_weight`/`prefetch hint`/`weight_tensor` 等全链路零改动（已核实 `single()` 只借指针不拷贝）。

## 3. 安全与限制（P0 已知边界）

- **共享对象守卫**：流送层对象若同时被常驻权重引用 → 显式抛错（当前模型无此情况）。
- **speculative/vision**：offload 与 speculative 后端互斥（校验拒绝）；vision 权重恒常驻。
- **KV auto**：槽位 DeviceBuffer（9B ≈170MB）不计入 `resolve_kv_capacity` 的显存核算，
  27B（≈230MB 槽位）建议显式 `--kv-capacity` 或留足 headroom。
- **`next_projection_hints`**：指向下一层槽位（dense 模型不消费该 hint）——MoE offload 属 P2 范畴。
- **性能**：P0 同步流送 + 页缓存源，27B 预期 decode ≈1 tok/s（见方案 §2.4）；P1 双缓冲
  + `PinnedHostBuffer` 源是下一步。

## 4. 冒烟命令

```powershell
ninfer.exe <artifact> --messages <utf8.json> --max-new 48 --greedy --no-thinking `
    --kv-capacity auto --offload-ratio 0.7
# 验收：与不加 --offload-ratio 的输出逐字一致；gpu weights used 显著下降
```

## 5. 27B 路径（下载与转换并行中）

- 官方 `Qwen/Qwen3.8-27B` safetensors（18 分片 ≈55.6GB）hf-mirror 4 路并行下载中 → `D:\models\Qwen3.8-27B\`。
- 27B 几何与 3.6-27B 相同（64 层 hidden 5120）→ op 层零改动；转换 recipe 需在
  `official_recipes.py` 注册 `qwen3_8_27b`（照 `_dense_groupwise(Q6)` 名字驱动）。
- 本机 RAM 31.7GB：offload 流送部分 ≈11.6GB + 系统，勉强可行；pinned 转换在 P1。
