# 27B offload：双缓冲被写坏的单缓冲 —— 定位、修复与带宽天花板

- 日期：2026-09-17
- 硬件：RTX 4060 Laptop 8GB（sm_89）+ i9-13900H + 2×16GB DDR5-5200，RAM 31.7GB
- 产物：`D:\deps\models\qwen38-27b-q4v2.ninfer`（16.14GB）、`qwen38-27b-q4v3mtp.ninfer`（16.59GB）
- 目标：为 Phase 3（全量 streamed linear 改走 CPU）做前置测量，实测"GPU 侧单 token 时间"这一截距

---

## 1. 结论摘要

| 项 | 结果 |
|---|---|
| ★★ 发现缺陷 | `ensure_layer` 的槽位等待写成"等上一层计算"，**双缓冲被彻底废掉，整条流水线完全串行** |
| 修复效果 | 有效上传速率 **10.4~10.9 → 11.5~11.9 GiB/s**；decode 各 ratio 一致 **−8%~−11%** |
| 修复后状态 | 引擎实测速率与独立探针的 pinned 裸上传速率（**11.75 GiB/s**）逐点吻合 ⇒ **传输已是唯一瓶颈，全部层计算被隐藏** |
| 被证伪的假设 | ① pin 失败走 staged copy（探针实测可锁 **13.25 GiB**）；② pageable 无法重叠（实测 pinned/pageable 都能与并发 kernel 重叠，delta ≤4.3%） |
| 新发现约束 | 本机 `cudaHostRegister` 上限 **≈13.25 GiB**（ratio 1.0 需 12.5 GiB，余量极小） |
| 对 Phase 3 的意义 | 流送侧没有可再挖的调度空间了；唯一剩下的杠杆是**不搬字节**，即 CPU 计算。当前最好 1.03 s/token，CPU 路径预估 ~0.345 s/token ⇒ **约 3×** |

---

## 2. 测量方法：用 ratio 扫描分离"流送"与"非流送"

`--offload-ratio r` 是**流送到 host 的比例**，因此流送字节 ∝ r，而"不随 r 变化的部分"
（GPU 计算、每 token 固定开销）就是截距。4 个 ratio 各 32 token、`--greedy`、同批次：

```
decode_ms_per_token(r) = a + b·r
  b = 层字节 / 有效拷贝带宽
  a = GPU 侧不随流送量变化的成本  <-- 这正是 Phase 3 前必须量出的截距
```

### 2.1 修复前

| ratio | decode tok/s | ms/token | GPU weights | H2D |
|---|---|---|---|---|
| 0.80 | 1.0 | 964 | 5.06 GiB | 5.06 GiB |
| 0.90 | 0.9 | 1072 | 3.89 GiB | 3.89 GiB |
| 0.95 | 0.9 | 1081 | 3.30 GiB | 3.30 GiB |
| 0.98 | 0.9 | 1131 | 2.91 GiB | 2.91 GiB |

拟合：**b ≈ 871 ms/ratio，a ≈ 271 ms/token**。
但"总时间 = Σ流送 + Σ计算"（完全串行）的预测与实测逐点吻合，说明那个"截距"其实是
**被串行化的层计算**，不是独立的固定开销。

### 2.2 关键一步：用引擎自报的驻留字节反推真实流送量

`ratio` → 实际流送量是**按层量化的阶梯函数**（`first_streamed` 按字节预算选层），
所以 4 点线性拟合的斜率/截距都不可信。改用引擎日志里的 `gpu weights used`：

```
layer_bytes  = 12.505 GiB（64 层，由 artifact 目录 JSON 精确统计）
resident(r)  = C + (1−r)·layer_bytes        C = 2.57 GiB（embedding+head+norm，恒驻留）
streamed(r)  = layer_bytes + C − resident(r)
```

代入实测的 `gpu weights used`：

| ratio | 流送 GiB | 修复前 ms/tok | 前·有效速率 | 修复后 ms/tok | 后·有效速率 |
|---|---|---|---|---|---|
| 0.80 | 10.02 | 964 | 10.39 GiB/s | 873 | **11.48 GiB/s** |
| 0.90 | 11.19 | 1072 | 10.44 | 953 | **11.74** |
| 0.95 | 11.78 | 1081 | 10.90 | 991 | **11.89** |
| 0.98 | 12.17 | 1131 | 10.76 | 1033 | **11.78** |

（ms/token = `model elapsed` 减去 prefill 时间；prefill 由日志的 `prefill speed` × `prompt tokens` 求得。）

---

## 3. ★★ 根因：双缓冲的等待条件指错了层

`WeightStreamScheduler` 的设计（头文件注释与 `streamed_bytes = 2 * slot_stride` 都印证）
是**两个槽位轮转**：第 L 层上传到槽 `(L−first)%2`，而该槽上一次的读者是**第 L−2 层**，
因此上传只需等 L−2 计算完，就能与 L−1 的计算重叠。

但实现里只有一个共享事件，且记录在**当前**计算流的尾部：

```cpp
// 修复前
CUDA_CHECK(cudaEventRecord(compute_event_, compute_stream));   // 尾部 = compute(L-1) 结束
CUDA_CHECK(cudaStreamWaitEvent(transfer_stream_, compute_event_, 0));
// ... 上传 L ...
CUDA_CHECK(cudaStreamWaitEvent(compute_stream, copy_event_, 0));
```

`copy(L)` 因此被钉在 `compute(L-1)` 之后，`compute(L)` 又在 `copy(L)` 之后 ——
**每次"拷贝↔计算"严格交替，两个槽位形同虚设**。这解释了 2.1 节"总时间 = Σ流送 + Σ计算"
的精确吻合，也解释了有效速率被压在 ~10.6 GiB/s（DMA 引擎在每次计算期间空转）。

### 3.0 代码考古：这个过严等待本来是给单槽修的

原注释写的是"the slot is **shared between streamed layers**: hold the uploads back until every
kernel enqueued so far (the previous layer's compute) has finished reading the slot" ——
"槽位被相邻层共享"正是**单槽**的描述。而分配侧（`streamed_bytes = 2 * slot_stride`、
`parity_offset = (layer − first) % 2 * slot_stride`）早已是**双槽**。

与工作日志对照可以还原时间线：

1. 最初是单缓冲同步流送（decode 0.9 tok/s）；
2. 发现 P0 竞态（`copies N` 与 `kernels N-1` 无同步）→ 用"上传等上一层计算"修掉，**对单槽正确**；
3. P1 引入双槽轮转（显存 +224 MB）以换取重叠，**但等待条件没有同步放宽**。

所以本次修复不是推翻旧结论，而是**把 P1 的双缓冲真正接通**。教训不是"改错了"，而是
**分配策略变了以后，与之配套的同步条件必须一起复核** —— 尤其当旧注释描述的是旧策略时，
注释会替错误的代码打掩护。

### 3.1 修复

按槽位奇偶各持一个事件，并且**提前一拍布防**：本次调用记录的事件，其尾部是"上一层结束"，
正好是**下下层的上传**需要等的量。

```cpp
const std::size_t parity = (index - spec_.first_streamed_layer) % 2;

if (index == spec_.first_streamed_layer) {
    // 跨 pass 边界（新的 decode step / 下一个 prefill chunk）：槽位被上一 pass 复用，
    // 上一 pass 布防的事件不保证晚于它 —— 在这里按当前尾部重新布防。这是唯一必须串行化的点。
    CUDA_CHECK(cudaEventRecord(compute_event_[parity], compute_stream));
}
CUDA_CHECK(cudaStreamWaitEvent(transfer_stream_, compute_event_[parity], 0));
CUDA_CHECK(cudaEventRecord(compute_event_[parity ^ 1u], compute_stream));
// ... 拷贝 ...
CUDA_CHECK(cudaEventRecord(copy_event_, transfer_stream_));
CUDA_CHECK(cudaStreamWaitEvent(compute_stream, copy_event_, 0));
```

- 不变式：`upload(L)` 只等 **`compute(L−2)`**。
- 靠"调用 L 记录 `parity^1`"实现：调用时刻的尾部是 `compute(L−1)` 结束，而
  `(L+2)%2 == L%2`，所以这次记录恰好服务于 L+2 的等待。
- 首次调用的 `compute_event_[parity]` 未记录过 → `cudaStreamWaitEvent` 视为已完成，不等待。
- pass 边界外**每层都重叠**；边界处只串行一层，代价可忽略（1/64）。

### 3.2 效果（同批次对照，32 token）

| ratio | 修复前 ms/tok | 修复后 ms/tok | 变化 |
|---|---|---|---|
| 0.80 | 964 | 873 | −9.4% |
| 0.90 | 1072 | 953 | −11.1% |
| 0.95 | 1081 | 991 | −8.3% |
| 0.98 | 1131 | 1033 | −8.7% |

有效速率从 10.4~10.9 GiB/s 抬到 11.5~11.9 GiB/s，**并且不再随 ratio 漂移** ——
这正是"瓶颈从调度变成链路"的签名。

### 3.3 正确性验证（三重）

槽位等待放宽是一处**内存序**改动：等错方向就是设备端数据竞争（第 L 层的上传覆盖第 L−2 层仍在读的半区），
表现为权重被破坏、输出乱码。三个观测量：

| 判据 | 结果 | 说明 |
|---|---|---|
| Greedy 逐字可复现 | ✅ 两次运行输出**逐字节相同**（271 字符） | 竞态会让两次结果分歧 |
| 输出连贯性 | ✅ 英文/中文均为正常文本 | 权重被破坏会立刻变成乱码 |
| **MTP 接受率** | ✅ 两次均为 **35.3% / 2.04 tok/round**，与修复前基线**完全相同** | 最强判据：verify 通道（T=4）被破坏时接受率会坍塌到 **0.0% / 1.00**（Phase 1c 的布局缺陷就是这个签名） |

MTP 吞吐：修复后 2.0 tok/s（27.3s/48tok），修复前基线 2.1（26.4s）。
差异 3.4%，与 greedy 两次运行的自身离散（50.8s vs 50.0s，1.6%）同量级 ——
**MTP 一轮里传输只占 ~0.96s，另有约 0.2s 是串行的非层工作**（draft 层 + head + 采样 + logits 回读），
这部分不参与重叠，因此重叠修复在 MTP 上收益很小。

> ⚠️ **已知残留（已量化，未优化）**：pass 边界的重新布防让每个 pass 的第一层上传等待
> "上一 pass 整体排空"，比理论所需多等**一层**（≈15 ms/pass）。
> 只有当 pass 的流送层数为奇数时才真正必要（此时最后一个槽位 0 的读者是末层本身）。
> 收益 ≈1.5~2%，需要新增一个"pass 结束"钩子才能消除，暂不引入。

---

## 4. 两个被证伪的假设（独立探针，不碰引擎代码）

`.workbuddy/tmp/pin_probe.cu`（`devlink.py` 直接 nvcc 编译链接）：

### 4.1 `cudaHostRegister` 到底能不能锁住大块？

```
cudaHostRegister, ordinary heap blocks（memset 触页后注册）：
   0.25 GiB  ok
   1.00 GiB  ok
   4.00 GiB  ok
   8.00 GiB  ok
   4.00 GiB  FAILED: out of memory
   total registered: 13.25 GiB
```

**能锁**。所以"pin 失败静默退回 staged copy"不是本次瓶颈的成因。
但暴露了一条此前没记录的真实约束：**本机可锁定量 ≈13.25 GiB**
（31.7GB RAM 上，WDDM 把锁定内存也计入 GPU 共享内存预算）。
ratio 1.0 需要锁 12.5 GiB —— **余量不到 6%**，因此
`--offload-ratio` 不能无限调高，且 ratio 与 `--host-embedding/--host-output-head` 叠加时要留意。

### 4.2 pinned vs pageable，以及能否与 kernel 重叠

```
upload 32 × 200 MiB（每步一层）:
  registered alone 0.532 s (11.75 GiB/s)   +并发读 bound kernel 0.533 s (11.72)  delta +0.2%
  pageable   alone 0.567 s (11.02 GiB/s)   +并发读 bound kernel 0.592 s (10.56)  delta +4.3%
```

- **两条路都能与并发 kernel 重叠**，"pageable 无法重叠"的假设不成立。
- pinned 只比 pageable 快 **~7%**。pin 仍值得保留（免费的 7%），但它不是 2× 级别的杠杆。
- 裸上传上限 **11.75 GiB/s**（≈ PCIe 4.0 x8 有效值），与 3.2 节引擎实测 11.5~11.9 GiB/s 吻合。

### 4.3 修复后的残差：传输=时间

以 11.75 GiB/s 预测上传时间，与实测 ms/token 相减：

| ratio | 预测上传 | 实测 | 残差 |
|---|---|---|---|
| 0.80 | 853 ms | 873 | +20 ms |
| 0.90 | 952 | 953 | +1 |
| 0.95 | 1003 | 991 | −12 |
| 0.98 | 1036 | 1033 | −3 |

残差在 ±20 ms 内（±2% 测量噪声）。**修复后 decode 时间 ≈ 流送字节 ÷ 链路速率，
没有可测的额外截距** —— 所有层计算都被隐藏在拷贝之下。

⚠️ **方法学提醒**：不要用"4 点拟合的截距"当截距。`ratio` 与真实流送量之间是按层量化的阶梯，
4 个点跨越的层数很少，拟合出的截距会被阶梯噪声放大 10 倍以上（本例拟合得 184 ms，
而按引擎自报字节算出的真实残差是 ~0）。

---

## 5. 对 Phase 3 的意义（比修复前更强了）

修复前以为"截距 271 ms 是 GPU 侧地板，Phase 3 只能到 ~0.35+0.27 s"。
修复后这个地板基本消失，结论反而更干净：

> **流送路径已经贴着链路天花板，没有任何调度空间可挖。**
> 1.03 s/token @ ratio 0.98 里，~1.03 s 全是"把 12.2 GiB 搬过 PCIe"。
> 要更快，只剩一个方向：**不搬这些字节** —— 即 Phase 3 的 CPU 计算。

| 方案 | 每 token | 说明 |
|---|---|---|
| 修复后流送（ratio 0.98） | **1.03 s** | 已到 PCIe 上限 |
| 修复后流送（ratio 0.80） | 0.873 s | 但 GPU 只能装 5.06 GiB |
| CPU 计算（Phase 1 实测外推） | **~0.345 s** | 145 GFLOP/s vs ~50 GFLOP/token |
| 预期收益 | **≈3×** | 且必须关 MTP（CPU 是算力受限，见 §4.3 of phase1 文档） |

---

## 6. Phase 3 op 清单（按流送字节排序）

用 `.workbuddy/tmp/artifact_bytes.py` 直接解析 artifact 目录 JSON
（binding 的 `parts[].range` 是**元素**范围，须乘对象的 encoded-size-per-element）：

每层 **0.195 GiB**（attention 层 0.1977、GDN 层 0.1885 GiB），64 层共 12.505 GiB。

| op（消费者） | 占流送字节 | GiB/层 | 覆盖的角色 |
|---|---|---|---|
| **`linear_swiglu`**（gate+up 融合） | **45.1%** | 0.088 | mlp/gate, mlp/up |
| **`linear_add`**（残差融合） | **28.5%** | 0.056 | mlp/down, gdn/output |
| `gdn_input_proj`（融合/配对） | 9.4% | 0.018 | gdn/query,key,value,z |
| `gdn_input_proj`（配对 Q8） | 9.4% | 0.018 | gdn 的另一半 |
| `attn_input_proj`（融合 qk+vg） | 5.2% | 0.010 | attention/query,key,value,gate |
| **`linear`** ✅ 已 host 化 | **2.0%** | 0.004 | gdn/z 单父形态 |
| `gdn_gating_proj` | 0.4% | 0.001 | gdn/a,b_projection |

格式分布（全 artifact）：`q4 10.251 GiB / q8 2.516 / q5 2.205 / bf16 0.049`。
CPU 内核（`ops::cpu::rowsplit_gemm`）已覆盖 q4/q5/q6/q8 的 RowSplit，**格式上无缺口**。

**要点**：`linear_swiglu` + `linear_add` = **73.6%**，而已经打通的 `linear` 只占 2.0%。
Phase 3 的性价比顺序就是这两个 op。

---

## 7. Phase 3 待办（更新版）

1. **`ensure_layer` 语义切换**：新增开关（建议 `--host-linear`）后，流送层不再上传；
   `spec.weight_objects[id]` 留空 ⇒ `load.cpp:385-396` 的重指向循环自然跳过 ⇒
   `bind_view` 保留 **host parent**；`spec.enabled=false` ⇒ 不分配设备槽位。
   与 `--offload-ratio` 正交：ratio 仍决定"哪些层驻留 GPU"，其余层**在 CPU 上算**。
2. **`linear_swiglu` host 路径**（45.1%）：`rowsplit_gemm` 到 host 暂存 [2M,T] bf16，
   host 端做 `SiLU(gate)·up` → 一次 H2D。权重是单一 Weight（gate/up 在同一父对象内连续）。
3. **`linear_add` host 路径**（28.5%）：读回 residual（D2H）→ `rowsplit_gemm` → host 相加 → H2D。
4. `gdn_input_proj` / `attn_input_proj` 的配对形态（18.8%+5.2%）：建议复用其已有的
   "independent" 路线，即逐个 part 走 host `linear`，conv/拼接仍留 GPU。
5. **必须关 MTP**（CPU 算力受限，MTP 在 CPU 上慢 2.4×）。
6. 预期 ≈0.345 s/token ≈ **2.9 tok/s**（当前最好 1.03 s/token）；实测后重估。
7. `ops::linear_pair`（Q8 [1024,k] 配对）如需覆盖 GDN k/v，需补 host 路径 + 补 k=4096（9B）。

---

## 8. 本次顺带修正的过期注释

| 位置 | 原文（错） | 现文 |
|---|---|---|
| `ops/cpu/rowsplit_gemm.h` | `out[n][tokens] = ...`，激活 `[k][tokens]` | 明确 token-major：`out[token*n + row]`、激活 `activation[token*k + column]` |
| `ops/cpu/rowsplit_gemm.cpp` staging 注释 | "设备路径给的是 `[k][tokens]`，CPU 想要 token-major" | 两边都是 token-major，staging 只为 bf16→fp32 一次扩容 |
| `offload_stream.h` `ensure_layer` 注释 | 未说明等待的是哪一层 | 明确"只等到上一次使用该槽位的层（即 L−2）" |

---

## 9. 复现命令

```powershell
# ratio 扫描（4 点，各 32 token，--greedy，同批次）
powershell -NoProfile -ExecutionPolicy Bypass -File .workbuddy\tmp\sweep_ratio.ps1
# -> .workbuddy\tmp\sweep_ratio_result.txt

# 正确性验证：greedy 两次逐字一致 + MTP 接受率对照 35.3%/2.04
powershell -NoProfile -ExecutionPolicy Bypass -File .workbuddy\tmp\verify_offload.ps1

# pin / 重叠探针（两种模式分开跑，失败会污染进程内的 pinned 记账）
python .workbuddy\tmp\devlink.py .workbuddy\tmp\pin_probe.cu .workbuddy\tmp\pin_probe.exe
.\.workbuddy\tmp\pin_probe.exe register
.\.workbuddy\tmp\pin_probe.exe copy

# artifact 字节构成
python .workbuddy\tmp\artifact_bytes.py D:\deps\models\qwen38-27b-q4v2.ninfer
```
