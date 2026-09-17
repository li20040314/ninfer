# P0-b：embedding 移出显存（host zero-copy gather）

日期：2026-09-17
分支状态：**已交付**（CLI + serve 路径均已构建、双路正确性验证通过；实测 decode 1.8→2.0 tok/s，
+11.1%，见 §7）。
前置：`rtx4060-27b-offload-speedup-plan-2026-09-16.md`（P0/A/B 已交付）、
`rtx4060-27b-e2-cpu-gemv-decision-2026-09-17.md`（E2 判决：不叠 MTP）。

---

## 1. 动机

27B 的 offload 里，**只有 text layer 参与流送**，而 `token_embedding` 与 `output_head` 是
"全局参数"，永远驻留显存：

```
WeightOffloadSpec 注释（offload_stream.h:35）
  "Earlier layers, the embedding, the output head, and the final norm stay device resident."
```

实测它们的真实占用（`tools.artifact.inspect`，权威）：

| 参数 | 格式 | layout | 字节 | 说明 |
|---|---|---|---|---|
| `text/token_embedding` | `q8_g32_fp16` | `row_split_k128_v1` | **1,350,860,800**（1.258 GiB） | 248320 × 5120 |
| `text/output_head` | `q8_g32_fp16` | `row_split_k128_v1` | **1,350,860,800**（1.258 GiB） | 248320 × 5120 |
| 层权重（q4+q5） | — | — | 22,649,241,600（**21.094 GiB**） | 64 层，每层 340.6–358.3 MB |

两者都不流送，却各占 1.258 GiB。**而两者被访问的方式完全不同**：

- **embedding 是稀疏行 gather**：每个 decode step 只读 T 行（T=1~8），行宽 5440 B。
- **output head 是 dense**：每步都要读完整张矩阵。

这个差别决定了两者的解法不同。本文只做 embedding。

## 2. 前置实测：host 权重能否被 kernel 直接读

工具：`.workbuddy/tmp/host_weight_probe.cu`（q8_g32 / row_split 真实布局，真实尺寸
248320×5120，1.258 GiB 表）。

**方法学要点（踩过的坑）**：

1. **必须每变体一个进程**。非法访问会变成 **sticky context error**，`cudaGetLastError()`
   清不掉 —— 第一个变体一挂，同进程后续所有测量都在报同一个旧错误。
2. **重复读同一批行会被 24 MiB L2 吃掉**。必须每次 launch 走 id 池的不同切片，否则测的是
   L2 不是 PCIe。

### 结果

| 变体 | 正确性（vs CPU 参考） | 稀疏 1 行 | 稀疏 256 行 | dense 全表 1.258 GiB |
|---|---|---|---|---|
| `device`（现状） | `2.63e-06` OK | **9.47 μs** | 13.6 μs | 8.3 ms（162 GB/s） |
| **`cudaHostAlloc`（pinned）** | **`2.63e-06` OK** | **8.51 μs** | 159.6 μs | **162.5 ms（8.3 GB/s）** |
| `malloc` + `cudaHostRegister` | ❌ illegal memory access | — | — | — |
| `malloc`（pageable） | ❌ illegal memory access | — | — | — |

### 三条可执行结论

1. **pinned host 内存可被 kernel 正确读**，且**稀疏读取甚至比 device 略快**
   （8.51 vs 9.47 μs；两侧都是 kernel 启动开销主导）。→ embedding 移位**近乎免费**。
2. **output head 不能用 zero-copy**：dense 只有 8.3 GB/s，全表 162 ms/步，而它现在在
   device 上只要 8.3 ms。它需要 CPU GEMV，是另一件事（见 §8）。
3. ★ **`cudaHostRegister` 在本机对大块不可用**：它返回成功，然后第一次 device 读就非法。
   而**引擎现有的 offload page-lock 正是用它**（`offload_stream.cpp:44`，
   "best-effort"）—— 这条路径在本机大概率一直没生效，H2D 走的是 WDDM staged copy。
   待验证（§8）。

## 3. 设计

### 3.1 为什么不是"CPU 做 lookup"

decode 走 CUDA Graph capture（`core/decode_graph.cpp:63`）。CPU 参与计算不能在 capture 内
发生。**zero-copy 把 kernel 的参数换成一个 host 指针，图完全不受影响** —— 这是本方案相对
E2（CPU GEMV）的根本优势。

### 3.2 机制：`residency = Host` + page-locked placement

引擎里已经有一条完整的"字节在 host、view 指向 host"的路：

```
PendingWeight.reference.residency = Residency::Host
  → artifact::bind_view() 选 host_parent() 而不是 device_parent()   (views.cpp:14)
  → Weight.qdata / scales 就是 host 指针
  → embed_gather kernel 直接解引用（无中间拷贝）                     (embed_gather.cu:34)
```

缺的只有一件事：**这块 host 内存必须是 page-locked**。所以新增 `HostPlacement.page_locked`，
在 materializer 里用既有的 `PinnedHostBuffer`（`core/arena.h:93`，`cudaHostAlloc` 包装）分配。

### 3.3 腾出的显存返还给层

`ratio` 的语义是"流送到 host 的比例"，`target_resident = (1-ratio) × layer_bytes`。
如果只是把 embedding 挪走而不动 ratio，腾出的 1.258 GiB 会**闲置**。所以：

```cpp
target_resident = (1 - ratio) × total_layer_bytes + host_only_bytes;
```

注意 **device arena 总大小不变**：原来 `层 4.219 + embed 1.258 + head 1.258`，现在
`层 5.477 + head 1.258`，都是 6.735 GiB。**这消除了 OOM 风险** —— 只是把闲置的显存
换成了"少流一层"。

## 4. 改动清单

| 文件 | 改动 |
|---|---|
| `src/artifact/materializer.h` | `HostPlacement.page_locked`；`ObjectStorage.host_pinned` |
| `src/artifact/materializer.cpp` | page_locked 走 `PinnedHostBuffer`；`host_bytes()` 支持两种存储 |
| `src/models/qwen3_5/load.cpp` | `host_only_objects` 集合；`target_resident` 返还；device→host 时标 `page_locked` |
| `src/models/load_options.h` | `LoadOptions.host_embedding` |
| `include/ninfer/types.h` | `EngineOptions.host_embedding` |
| `apps/cli/options.{h,cpp}`、`apps/cli/main.cpp` | `--host-embedding` |
| `src/serve/serve_options.{h,cpp}`、`serve/generation_service.cpp` | 同上（serve 路径） |

**执行层零改动** —— kernel 拿到的只是另一个指针。这是本方案最重要的工程性质。

## 5. 使用

```powershell
ninfer.exe <model.ninfer> --offload-ratio 0.9 --host-embedding ...
```

`--host-embedding` 只在 `--offload-ratio` 非零时有效（收益来自"腾显存给层"，没有 offload
就无从返还）。

## 6. 预期收益（按实测字节）

| 项 | 现状（ratio 0.9） | +`--host-embedding` |
|---|---|---|
| 层驻留 | 2.109 GiB（≈7 层） | 3.367 GiB（≈10 层，实际受层粒度取整约束，见 §7.2） |
| 层流送 | 18.98 GiB | **17.73 GiB（−6.6%）** |
| device 权重总计 | 6.735 GiB | 6.735 GiB（预期不变，实测见 §7.2 的取整偏差） |
| embedding 每步成本 | 0 | +8.5 μs（可忽略） |

## 7. 实测

同一产物 `qwen38-27b-q4v3mtp.ninfer`、同一 seed、同一 `--offload-ratio 0.9`、同一 `--spec mtp
--draft-tokens 3`，`run-27b.ps1 bench`（max-new 512，实收 201 token），加/不加 `--host-embedding`
各跑一次。两侧 MTP 接受率 **35.0%**、接受长度 **2.05 tok/round 完全一致**，可比性充分。

| 指标 | 基线 | +`--host-embedding` | Δ |
|---|---|---|---|
| decode | 1.8 tok/s | **2.0 tok/s** | **+11.1%** |
| prefill | 5.5 tok/s | **7.3 tok/s** | +32.7%（见下） |
| model elapsed（201 token） | 1m 55.6s | **1m 45.6s** | **−8.7%** |
| gpu weights used | 4.31 GiB | 4.22 GiB | −0.09 GiB |

### 7.1 正确性（两道闸门都过）

| 路径 | 命令 | 输出 |
|---|---|---|
| CLI | `ninfer.exe … --messages … --host-embedding` | 「太阳辐射使海洋、湖泊和河流中的水蒸发升空，同时植物通过蒸腾作用也向大气释放水汽。这些水汽在高空遇冷凝结成云，随后以雨、雪等降水的形式重新落回地球表面。降水渗入地下或汇入江河湖海，最终再次进入……」 |
| serve | `ninfer-serve` @8091 → `POST /v1/chat/completions` | 「光合作用是植物、藻类和某些细菌利用光能，将二氧化碳和水转化为有机物（如葡萄糖）并释放氧气的生物化学过程。这一过程不仅为生物圈提供了主要的能量来源和氧气，也是维持地球生态平衡的关键……」 |

两条路径语义连贯、无乱码 —— 说明 host 侧的 embedding 表被 kernel 正确读到（指针错一位就会
立刻退化成乱码或崩溃）。serve 侧为新编译产物（见 §4 改动已进入 `ninfer-serve.exe`）。

### 7.2 为什么 gpu weights 只降 0.09 GiB

§6 预期「device 权重总计不变」，实测基本吻合但有小幅下降，原因是**层的分配粒度**：

- 腾出的额度 = embedding 的 1.258 GiB；
- 规划器把它全部还给常驻层，但单层 340~358 MB，装不满整数层 → 多驻留 **3 层 ≈ 1.02 GiB**，
  余下约 0.24 GiB 因取整闲置（表现在 `free after weights` 上升）。

所以**显存侧的净变化很小（−0.09 GiB），真正的收益在流送侧**：层流送量
18.98 → 17.73 GiB（−6.6%）。decode 的 +11.1% 主要来自这里。

### 7.3 prefill 数字的置信度

bench 用的提示词很短，prefill 只覆盖几十个 token，`prefill speed` 本身噪声大，
+32.7% 这个幅度**不要当作可靠结论**。可靠的是 decode（+11.1%）与 model elapsed（−8.7%）——
两者方向一致、量级吻合流送量的 −6.6%。

## 8. 已知边界与后续

- **只做了 embedding**。output head 需要 CPU GEMV（Phase 1），依据见 §2 结论 2：
  同一张 1.258 GiB 的表，零拷贝读单行是 8.5 μs（可忽略），但整表稠密读是 **162.5 ms/step**
  （8.3 GB/s，被 PCIe 有效带宽锁死）——output head 每步都要扫全表，所以这条路对它不成立。
- ✅ **`cudaHostRegister` 在本机对大块内存失效已确认**：probe 返回成功，但 kernel 首次读取即
  illegal memory access（pageable 同样）。engine 现有 offload 的 page-lock（`offload_stream.cpp`
  的 "best-effort"）因此**大概率从未真正生效**，H2D 一直走 WDDM staged copy。改用
  `cudaHostAlloc`（`PinnedHostBuffer`）后既有收益空间，独立于本方案，值得单独立项验证。
- **`--host-embedding` 只在 `--offload-ratio` 非零时有意义**：收益来自"腾显存给层"，没有 offload
  就无从返还（代码里也据此只在 offload 规划路径生效）。
- **ratio 未做自动换算**：用户传的 ratio 语义不变，引擎负责返还。若**同时**手动调低 ratio 会
  双重收益（也更可能 OOM）。
- **MTP/draft 共用同一 `WeightId`**（`load/mtp.cpp:13`、`load/dflash.cpp:18`），
  所以一处标 Host 即全部生效 —— 已覆盖，无需重复改。

