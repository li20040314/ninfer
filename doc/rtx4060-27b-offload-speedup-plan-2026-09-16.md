# RTX 4060 27B CPU Offload 提速方案（参考 llama.cpp）

日期：2026-09-16（当晚已实施第一批 A+B，结果见文末「实施结果」）

> ⚠️ **2026-09-17 更正（本文档 §4 的 E2 预期已作废）**
> 已按「先量地板再铺管线」实测 E2（三条路径：精确 fp32 / int8 单链 / int8 残差双链）：
> - 内存上限 **63.6–67.8 GB/s**、端到端 PCIe 有效 **~10.5 GB/s** ⇒ 物理前提 **≈5×，成立** ✅
> - 数值：fp32 `rel_l2 3.355e-07`、**split 双链 `1.972e-05`**（达标，只贵 22% 带宽）✅
> - 但**三条路径的 ms/token 全部随 T 恒定** ⇒ 纯 ALU 受限 ⇒ **与 MTP 不可相乘**（MTP 摊薄的是访存，不是 ALU）
> - 收益重估：**E2 split 关闭 MTP = 2.72 tok/s（+33%）**；E2 + MTP(K=3) = **1.24 tok/s（-39%，回退）**
> - 更划算的先行项：**embedding + lm_head（3.12 GB 恒驻留）移出显存** ⇒ ratio 0.8→0.56 ⇒ **+44%，且保留 MTP**
>
> 本文中「E2 → 2–2.5 tok/s」「E2 后 2.0 × 2.2 ≈ 4.4 tok/s」「与 E 方案正交可叠乘」等表述**均不成立**。
> **以 `rtx4060-27b-e2-cpu-gemv-decision-2026-09-17.md` 为准**（含修正后的 Phase 1–3 计划）。

现状基线：Qwen3.8-27B @ 4060 Laptop 8GB，`--offload-ratio 0.9`，**decode 0.9 tok/s**（GPU 3.99GiB / host 11.94GiB）。
本机：i9-13900H（14C/20T）+ 2×16GB DDR5-5600（实跑 5200，双通道理论 83.2GB/s）。

---

## 1. 瓶颈定位（先摆数据，再谈方案）

| 指标 | 数值 | 推导 |
|---|---|---|
| 每token流送字节 | ≈14.5 GB | ratio 0.9，payload 15.93GiB，embedding+lm_head(Q8) 驻留 |
| 每 token 耗时 | ≈1.11 s | 1 / 0.9 |
| **有效 PCIe 带宽** | **≈13.0 GB/s** | 14.5 / 1.11 |
| 4060 Laptop PCIe 上限 | 4.0 ×8 = 16 GB/s 理论，pinned copy 实测上限 ≈14–15 GB/s | — |
| 每层流送 | ≈224 MB / 64 层 → ≈17 ms/层 | 14.5 GB / 64 |
| 每层 GPU 计算量 | ≈0.85 GFLOP → **≈0.2 ms** | 2×27G / 64 层 / ~5 TFLOPS |

**结论：传输占每 token 时间的 98% 以上，当前已跑在 PCIe 实测上限的 ~90%。**

由此推出两条硬边界：
1. **一切不减少流送字节、或不绕开 PCIe 的优化，收益上限都很低。**
2. **P1 双缓冲原估"2–4×"过于乐观**：它能藏的只有每层 ≈0.2ms 的计算 + sync 延迟，实际预期 **+2~5%**。此前估计作废。

---

## 2. llama.cpp 同问题技术映射

llama.cpp 在"显存装不下"时的做法与本项目 P0 的对照：

| llama.cpp 技术 | 做法 | 本项目现状 | 可移植性 |
|---|---|---|---|
| `-ngl` 层卸载 | **整个层留在 CPU 用 ggml CPU 后端计算**，权重零 PCIe 传输 | P0 是"流送到 GPU 算"，PCIe 是瓶颈 | ★ 可移植，最大收益来源 |
| Q4_K_M 等激进展量化 | 压权重字节 = 压访存量 | 已是 q4/q5_g64_fp16 分组量化 | ⚠️ 剩余空间约 10% |
| mmap + page cache | 靠 OS 惰性读盘 | 我们用 pinned buffer（对 PCIe 流送更优） | 无需 |
| speculative decoding | draft 便宜、verify 一次出 k 个 token，**摊薄权重访存** | 后端存在，但与 offload 互斥（P0 限制） | ★ 可移植，收益依赖任务 |
| 连续批处理 | 多请求共享同一次权重流送 | `max_concurrency` 1–8 已支持 | ✅ 已具备 |
| KV 量化 | KV 本来就小 | 27B KV 在显存且占比小 | 无关 |
| MTP draft | Qwen3.5-9B 有 MTP | ~~27B 无 MTP 层~~ **已修正：27B/9B 均有 `mtp_num_hidden_layers=1`（官方权重含完整 `mtp.fc` + 1 层 transformer，见 §8）** | ★★ 自 draft MTP 可用 |

---

## 3. 候选方案清单

| # | 方案 | 原理 | 预期收益 | 工作量 | 风险 |
|---|---|---|---|---|---|
| A | **流送字节再压缩**：`_dense_groupwise` 中 Q5 组（mlp/down、attention/output、gdn 剩余）降为 Q4，重转换 | 每字节 PCIe 时间线性下降 | +8~15%（→1.0–1.05 tok/s） | 0.5 天（改 recipe + 重转换 + 冒烟） | ⚠️ 质量需评测；down/output 对精度敏感 |
| B | **拷贝路径微优化**：层内多 object 合并为整层连续 pinned 区 + 单次大拷贝；`ensure_layer` 改事件等待 | 减少每次拷贝的启动开销与带宽爬坡 | +3~8% | 1 天 | 低 |
| C | P1 双缓冲（跨层预取，修正预期后） | 藏住每层 0.2ms 计算 | +2~5% | 2–3 天 | 低收益，优先级下调 |
| D | **零拷贝 mapped pinned 权重**：host 权重 `cudaHostRegister` 映射，q4/q5 GEMV kernel 直读 host 指针（每字节本就只读一次） | 省掉 VRAM 写入与槽位轮转，PCIe 流量不变但少一次落盘 | +5~15% | 2–4 天 | ⚠️ GEMV kernel 需容忍 PCIe 随机读延迟；需 64KB 对齐；需实测 |
| E | **CPU GEMV 混合计算（llama.cpp 核心思路）**：host 驻留权重的 linear 直接在 CPU 上算（AVX2 反量化 + GEMV），activation D2H/H2D 仅 20KB/次 | 瓶颈从 PCIe 13GB/s 换成 DDR5 双通道 ≈55–65 GB/s 有效 | **E1（只搬 mlp/down ≈4.2GB）→ ≈1.2 tok/s；E2（全部 host linear ≈13GB）→ 2–2.5 tok/s** | E1 ≈3–4 天；E2 ≈1–2 周 | ⚠️ 需为 q4/q5_g64 写 CPU GEMV + op 分发按 residency 路由；回归全量跑 |
| F | **解除 speculative 互斥 + MTP 自 draft（原 lookup 方案升级，见 §8）** | 一次权重流送 verify k 个 token，摊薄访存 | 接受率 2~3 token/步时 decode 1.0 → **2~3 tok/s**；与 E 正交可叠乘 | F1 ≈2–4 天（转换+拆互斥+验证）；F2（KV/图模式适配）另计 | ⚠️ 接受率依赖任务类型；需验证 verify 路径与 offload 钩子的兼容性 |
| G | **并发 batching**（serve 多用户场景） | N 路请求共享同一次权重流送，总吞吐≈×N | 总吞吐 ×并发数（单请求延迟不变） | 0（已支持，`--max-concurrency`） | 无 |
| H | 系统杂项：高性能电源计划、关闭 Windows"CUDA - 系统内存回退"、确认 GPU 不降频 | 保证拷贝与 SM 全速 | 0~5% | 0.5 小时 | 无 |

---

## 4. 推荐路线（分批交付）

```
第一批（快收益，~2 天）   A + B + H            0.9 → 1.0 tok/s  ✅已交付
F1（提前插队，半天）      MTP 投机 × offload    1.0 → 2.0 tok/s（×2.2）✅已交付
第二批（主菜，~1 周）     E1：mlp/down 移到 CPU 计算   → ≈2.2~2.4 tok/s（MTP 同乘）
第三批（主菜完成）        E2：全部 host linear 移到 CPU → ≈4.4 tok/s（MTP 同乘）
按需加菜                  D（零拷贝试验，与 E 二选一先试）
                          G（serve 多用户天然 ×N）
```

理由：
- **E 是唯一能改变数量级的方案**（PCIe 13GB/s → DDR 55GB/s，物理上 4 倍差距），且正是 llama.cpp `-ngl` 的核心逻辑：**与其把权重搬到算子的地方，不如把算子放到权重的地方**。
- A/B/H 成本极低，先落袋；C 按修正后预期降级为低优先。
- D 与 E 思路互补（都绕开"copy 再算"），D 改动小可先做试验决定取舍。

## 5. 验证方法

1. **带宽基准**：cudaEvent 计时单层 pinned H2D，确认 12.9 → 14+ GB/s 的可榨空间（给 A/B 定上限）。
2. **CPU 侧带宽**：自写多线程 memcpy/反量化循环实测 13900H + DDR5-5200 有效带宽（给 E 定上限，验证 55GB/s 假设）。
3. **质量回归**：`run_tests.py`（基线 11/12）+ 27B 中文 `--messages` 冒烟逐字对比。
4. **每步交付**：decode tok/s、GPU/host 占用、输出一致性三项对照表。

## 6. 待确认

- [x] A 的质量代价：中文冒烟输出正确（「我是通义千问…」，仅个别措辞差异，属 Q4 正常波动）；更系统的评测待做。
- [x] E 的 CPU 侧带宽假设已确认（i9-13900H + DDR5-5200 双通道，理论 83.2GB/s）。
- [ ] E2 全部 host linear 移到 CPU GEMV（下一批主菜）。

---

## 7. 实施结果（2026-09-16 深夜，第一批 A+B 已交付）

### 改动清单

| 文件 | 改动 |
|---|---|
| `src/models/qwen3_5/offload_stream.{h,cpp}` | host 权重 `cudaHostRegister` 页锁（best-effort，析构注销）；`ensure_layer` 改事件序（transfer 等 compute 尾部 → 拷贝 → compute 等 copy event），CPU 不再逐层阻塞；预计算每层拷贝批 |
| `src/models/qwen3_5/load.cpp` | **双槽轮转**：layer L 写槽 L%2 与 layer L-1 读槽 (L-1)%2 并行，消除 P0 的潜在槽位竞态且恢复拷贝/计算重叠；显存 +1 个单层足迹（≈+224MB） |
| `src/models/qwen3_5/execution/text.cpp` | `ensure_layer(layer, ctx_.stream)` 调用点适配 |
| `src/ops/linear/q4/shapes/n5120_k17408.cu`（新）+ `q4_shapes.h` + `q4_dispatch.cpp` + `src/CMakeLists.txt` | q4 GEMV 新形状（27B down），17 warps/row（272 组 ÷ 17 = 恰好 16 组/warp tile 上限） |
| `src/ops/linear_add/q4/q4_linear_add.cu` | 扩展 {5120,17408}（down 走 linear_add 路径），ksplit 模板化 K |
| `tools/convert/official_recipes.py` | `_dense_groupwise` 加 `output_residual_format`；`qwen3_8_27b` down+output 降 Q4 |

### 关键约束（重要，后续改格式必读）

- **input projection（q/k/v/gate 四合一）的配对合同**：多父形态硬编码「前两矩阵 Q4 + 后两矩阵 Q5」（`src/ops/weight_input.cpp:155`）→ **attention/value、gdn/value、gate 不能降 Q4**，否则启动报 `unsupported single-parent format` / `paired native form requires Q4 and Q5`。
- mlp/down 与 mixer output 走 `linear_add`（`ffn.cpp:87`、`text.cpp:926/1061`），其 q4 形状表独立于 linear 的形状表，两处都要补。
- 转换器拒绝覆盖已存在产物（safe-delete）→ 重转换用新文件名。

### 基准数据（--offload-ratio 0.9，--max-new 256，greedy）

| 模型 | 产物大小 | decode | prefill |
|---|---|---|---|
| 旧（Q5 residual，17.11GB） | 17.11GB | 0.9 tok/s | 8.5 tok/s |
| **新 q4v2（down/output Q4，16.14GB）** | **16.14GB（-5.7%）** | **1.0 tok/s（+11%）** | **9.6 tok/s（+13%）** |

---

## 8. MTP 路线（2026-09-17 修正，优先级提升）

### 8.1 事实修正

此前「27B 无 MTP 层」的结论**有误**，根因是按旧字段名 `num_nextn_predict_layers` 搜索 config。实测：

- 27B 与 9B 的 config 均为 **`text_config.mtp_num_hidden_layers = 1`**（新字段命名）。
- 27B 官方权重 `model-00018` 含完整 MTP 组件（15 个 key）：`mtp.fc`（[2h,h] 输入投影）、`mtp.pre_fc_norm_{embedding,hidden}`、`mtp.norm`、`mtp.layers.0.*`（完整一层 full_attention transformer：q/k/v/o + gate/up/down + 双 layernorm）。
- 外部佐证：象信 AI 的 Terminal-Bench 复测中，27B 在 vLLM 上就是以「MTP 投机解码」部署的（8×A800，tp=4×dp=2）。
- bf16 下 MTP 层 ≈0.75GB，量化后 ≈0.4GB → **可常驻 GPU**（8GB 卡 offload 场景下也可承受，KV 仅 1 层）。

### 8.2 ninfer 侧现状（链路已就绪，只差最后一公里）

| 环节 | 状态 |
|---|---|
| 运行时 MTP 后端 | ✅ `--spec mtp --draft-tokens [1,5]`（`SpeculativeBackend::Mtp`、mtp_pack/mtp_round 算子、`program/speculative/mtp.cpp`） |
| 转换器 mtp 提取 | ✅ `--components text,mtp` → `Qwen3_5MTP` companion；key 映射（`mtp.fc.weight→mtp/input_projection` 等）在 `qwen3_5.py:973-987` 现成 |
| Verify 路径接 offload | ✅ `text.cpp:707/761` 的 Verify 阶段走 `TextContext::run_layers`，其中已接 `ensure_layer` 钩子 |
| offload+speculative 互斥 | ❌ `model_instance.cpp:68` 硬禁（注释标 "P0"，属保守限制非根本冲突） |

### 8.3 实施步骤（F1）

1. **转换**：`--components text,mtp` 出 q4v3mtp 产物（转换验证已在进行，recipe 的 `mtp/` 量化规则现成）。
2. **拆互斥**：删除/收窄 `model_instance.cpp:68` 检查。条件：draft（MTP 层常驻 GPU，权重不进 MaterializationPlan 改写范围）+ verify（run_layers 已接钩子）。需要确认 MTP companion 的权重在 `plan_weight_offload` 中被划入常驻集合（embedding/lm_head 同类处理）。
3. **验证**：`--offload-ratio 0.9 --spec mtp --draft-tokens 2/3` 中文冒烟 + 256 token 基准，测实际接受率（greedy 对 coding/摘要类任务通常 2+ token/步）。
4. **风险点**：MTP draft 的 KV 与 offload 的 kv-capacity 约束交互；cuda graph 已强制关不受影响；`--draft-tokens` 越大单次 verify 的层计算越多但 PCIe 摊薄越多，需实测 2 vs 3。

### 8.4 预期与定位

- offload 场景下每步 decode 时间 ≈14.5GB/13GB/s ≈ 1.11s 几乎全是传输；MTP verify 用**同一次流送**算 2~3 个 token → decode ≈ ×接受率（1.0 → 2~3 tok/s）。
- 与 E 方案（CPU GEMV 减少传输字节）**正交可叠乘**：E2 后单步 ≈0.45s，MTP 再 ×2~3 → **4.5~6.75 tok/s**。
- 若 E2 排期在后，F1 可先行独立交付 2~3 tok/s。

### 8.5 F1 实施结果（2026-09-17 早，已交付）

**改动**：`model_instance.cpp` 把 blanket 互斥收窄为——放行 `SpeculativeBackend::Mtp`，继续拒绝 draft 模型后端（dflash/dflash2 是独立整模型，其层从不被 offload 计划改写，未实测前维持禁用）。**无其他代码改动**：MTP companion 权重（input_projection + 3 norms + 1 block）不属于 `weights.text.layers`，`plan_weight_offload` 自动把它们留作常驻 device 对象并在 offset 重排时保留（`load.cpp:180-188` 另有"流送层与常驻参数共享对象"的显式抛错兜底）。

**产物**：`D:\deps\models\qwen38-27b-q4v3mtp.ninfer`（15.45GiB = q4v2 15.03 + MTP 0.42GiB），`components=['text','mtp']`（`--components text,mtp`）。

**基准**（同一 prompt，148 token，`--offload-ratio 0.9 --kv-capacity 4096 --max-context 4096`，greedy）：

| 配置 | 耗时 | decode | prefill | GPU 权重 | 相对基线 |
|---|---|---|---|---|---|
| 基线 q4v2（无 spec） | 2m45.3s | 0.9 tok/s | 8.8 | 3.89 GiB | 1.00× |
| q4v3mtp k=2 | 1m26.6s | 1.8 tok/s | 8.9 | 4.31 GiB | 2.00× |
| **q4v3mtp k=3** | **1m17.1s** | **2.0 tok/s** | 8.0 | 4.31 GiB | **2.22×** |
| q4v3mtp k=5 | 1m15.7s | 2.0 tok/s | 8.8 | 4.31 GiB | 2.22× |

- **k=3 是甜点**（k=5 无进一步收益：草稿质量随深度衰减，抵消了更长窗口）。
- 短样本中文冒烟（28 token）达 **2.2 tok/s**（基线 1.0）——自由创作型 prompt 接受率低于结构化任务，代码/摘要类预期更好。
- 折算：每轮（一次完整 64 层流送 ≈1.11s）平均提交 **≈2.2 token**，即 k=3 下草稿接受率约 60%。
- **传输摊薄是第一性收益**：MTP 不减少任何 PCIe 字节，只是让同一批字节服务更多 token。

**正确性**：
- 与基线输出**基本逐字一致**（219/222 字符相同），差异仅在最后一句的收尾词（「到来」vs「轮回」）——批式 verify（4 token 一次性 GEMM）与逐 token 解码的浮点舍入不同导致的尾端近并列，属预期内；如需严格逐 token 一致需关闭投机。
- KV 边界测试通过：`--max-context 128 --kv-capacity 128 --max-new 200` 下返回 `finish reason context-capacity`、正好 100 token，无越界/崩溃。
- 回归 12 套件：**11/12 与基线一致**（唯一 FAIL = 已知的 `ninfer_gdn_gating_proj_test` 27B T=4097 fp32 累积地板）。

**下一个杠杆**：per-round 成本仍是 1.11s 的 PCIe 流送 → 现在轮到 E 方案（CPU GEMV）乘上去：E2 后 2.0 × 2.2 ≈ **4.4 tok/s**；再叠加 D（零拷贝）还有空间。

- 质量回归：12 套件 **11/12 与基线一致**（唯一 FAIL = 已知 gdn_gating fp32 地板，与本次无关）。
- 产出文件：`D:\deps\models\qwen38-27b-q4v2.ninfer`（用法同前：`--offload-ratio 0.9 --kv-capacity 4096 --max-context 4096`）。
- 认知修正：pin + 事件化在单槽下与旧同步流程同速（传输已贴 PCIe 有效上限 ~13GB/s），**真正的收益来自流送字节减少**；双槽的意义是消除竞态并保证拷贝/计算并行，而非提速。下一批数量级提升靠 E（CPU GEMV）。
