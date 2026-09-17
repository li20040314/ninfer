# Phase 3 交付：`--host-linear` —— 流送层改走 CPU 收缩（2026-09-17）

## 1. 目标与结果

Phase 2 修复双缓冲后，offload 路径的瓶颈已贴死 PCIe 有效上限（11.75 GiB/s）：
每 pass 必须把 ~12.5 GiB 权重推过总线，decode ~0.9 tok/s。Phase 3 换思路——
**不再搬运权重，让权重留在 host 内存、收缩在 CPU 上算**，把传输本身消掉。

| 配置（27B q4v2 @ RTX 4060, 48 tok, greedy） | decode | prefill | 权重驻留 |
|---|---|---|---|
| `--offload-ratio 0.98`（流送基线） | 0.9 tok/s | 3.5 s | 2.91 GiB |
| `--offload-ratio 0.98 --host-linear`（锁页 arena） | **2.1 tok/s** | 8.5 s | 5.85 GiB |
| `--offload-ratio 0.95 --host-linear`（锁页 arena） | **1.9 tok/s** | 9.5 s | 6.15 GiB |

**decode 2.1~2.3×**。三组生成文本逐字节相同（中文提示词，流送与 host 路径数值一致）。

> ★ **锁页是硬前提，不是优化**（见 §2.5）：host 权重 arena 若用 pageable 内存，
> 在本机内存压力下会被持续换出/换入（实测硬缺页 10~20 万页/秒），decode 直接
> 跌回 0.9~1.0 tok/s——表面看像"host 路径没效果"，实为换页抖动。首次交付时
> 未锁页，隔日复测暴露此问题。

## 2. 设计

### 2.1 开关语义（与 `--offload-ratio` 正交组合）

- `--offload-ratio r` 不变：仍按字节比例选 resident/streamed 层分界。
- `--host-linear`：把"streamed 后缀"的**可 CPU 化权重**改为 host 驻留 +
  CPU 收缩，其余保持设备驻留；不分配槽位区、不建调度器（`spec.enabled=false`）。

### 2.2 ★ hostability 谓词（本轮核心修正）

初版把 streamed 层**全部**权重 `place_on_host`——这是 P0：
`gdn_input_proj`（18.8%）/`attn_input_proj`（5.2%）/`gdn_gating_proj`（0.4%）与各
norm 没有对应 CPU 算子，设备 kernel 拿到 host 指针直接 fault。

修法：按 `ModelWeights` 结构精确枚举有 CPU 路由的消费者：

| 权重 | 消费算子 | host 路由 |
|---|---|---|
| FFN gate+up（融合 [2M,K] 单 parent） | `ops::linear_swiglu` | `linear_swiglu_host`（SiLU 折入 host 侧，只回写 [M,T]） |
| FFN down | `ops::linear_add` | `linear_add_host`（读回 residual，fp32 相加折入写回） |
| mixer output（attention/GDN 皆然） | `ops::linear_add` | 同上 |

覆盖流送字节 45.1% + 28.5% = **73.6%**；其余 ~26% 保持设备驻留（这就是
host-linear 模式权重驻留 5.85 GiB **高于**流送 2.91 GiB 的原因——被免掉的是每
token 的 PCIe 流量，代价是这部分常驻）。

### 2.3 host 收缩实现（`src/ops/linear/host/linear_host.{h,cu}`）

- 判定：`weight_is_host_resident` —— `cudaPointerGetAttributes` 对
  `qdata` 查询，`Unregistered`/`Host` = host（老 runtime 拒答也按 host 处理并
  清掉挂起的 `cudaErrorInvalidValue`）。结果按指针缓存（decode 每 token 同序遍历
  同一批权重，摊成每权重一次查询）。
- 传输：激活按 token 行 D2H 读回（尊重 nb[1] 节距）→ 一次 stream sync →
  `cpu::rowsplit_gemm`（AVX2，token-major [T][N] 输出）→ 写回保持异步。
- 数值：SiLU 用 fp32 精确式（镜像 `math.cuh`），乘积一次性舍入 bf16；
  linear_add 用 fp32 相加、单次舍入——与设备 epilogue 一致，这是三组输出
  逐字节相同的原因。
- 暂存：thread_local 高水位复用；async 目的组可在 sync 前扩容，async 源组只能在
  sync 后动（防上一拍写回仍被读取）。

### 2.4 约束（在 `model_instance.cpp` 强制）

- `--host-linear` 必须配 `--offload-ratio`（否则报错，防止看起来像 3× 开关却无效果）。
- 与 speculative decoding 互斥：MTP verify 使 CPU 收缩量随 draft 长度线性增长，
  实测接受率 2.04 摊不平 2× 收缩成本。
- `spec.enabled=false` ⇒ 调度器不创建；执行层 `streamed_` 空指针防护已存在。

### 2.5 ★ host arena 必须锁页（`place_on_host(page_locked=true)`）

host-linear 模式下 9.2 GiB 权重进入 host 主存（pageable）。首次交付用了
`page_locked=false`，当时工作正常（1.9~2.0 tok/s）；隔日系统 commit 压力
升高（70.8/87.7 GB）后重测，decode 跌回 0.9~1.0。

- **归因手段**：`linear_host.cu` 加四桶累计计时（readback / gemm / epilogue /
  writeback，atexit 打 stderr）→ gemm 44.9~54.7s（基准推算应 ~20s）；
  `Get-Counter '\Memory\Pages Input/sec'` 运行期采样 → **10~20 万硬缺页/秒
  （≈400-800 MB/s 从页面文件读回），可用内存一度只剩 375 MB**——权重被
  换出后每 token 全量读回。
- **修复**：`load.cpp` 白名单分支 `place_on_host(page_locked=true)`
  （`PinnedHostBuffer` = `cudaHostAlloc` + 分配即锁；失败抛异常不静默降级）。
  探针已证本机可连续锁 13.25 GiB > 需求 ~9.2 GiB（WDDM 把锁定页计入 GPU
  共享内存预算，需与 `--gpu-memory-budget` 一起规划）。
- **效果**：gemm 54.7s → 19.5s（2.8×），decode 0.9 → 2.1 tok/s，且不再随
  会话间系统状态漂移。锁页不改数值——输出与未锁页版本逐字节一致。
- **教训**：任何"host 驻留 + 每 token 全量随机读"的内存都必须锁页；部署机
  commit 余量与锁定预算要一起核。CPU 基准（rowsplit_bench）测得 130-190
  GFLOP/s 而引擎内只有 ~40-50，差值全部是换页——**引擎内归因计时是发现
  这类问题的唯一可靠手段**。

## 3. 根因 / 修复 / 验证（本轮踩坑）

| 问题 | 根因 | 修复 | 验证 |
|---|---|---|---|
| 上轮会话中断导致编辑半落盘 | wrapper 只有 include 没有分支（linear_swiglu）、只有分支没有 include（linear_add） | 逐一读盘核对补齐 | 编译错误 C2039 指路 |
| ninja 编了 obj 不出 exe | `targets` 模式退出码藏在日志文件，首次构建 FAILED 后仍显示正常 | 每次构建后核对 `targets-89.log` 尾部 + exe mtime | `Linking CXX executable` 行 |
| P0 全量 host 会 fault | 无 CPU 路由的权重被 `place_on_host` | §2.2 谓词白名单 | smoke 三组输出逐字节相同 |
| ★ host arena 换页抖动（decode 2.0→0.9 漂移） | pageable 权重内存被系统换出，每 token 硬缺页 10-20 万次 | `place_on_host(page_locked=true)`（§2.5） | gemm 54.7→19.5s；`Get-Counter` 缺页采样归零；输出逐字节不变 |
| CPU 基准测得 130-190 GFLOP/s、引擎内仅 ~45 | 差值 = 换页（非内核质量问题） | 同上 | `linear_host.cu` 四桶计时交叉验证 |
| bash 沙箱看不到新 exe | 沙箱与真实 FS 脱同步（旧已知问题） | Glob/Python `os.path.getmtime` 交叉验证 | exe mtime 与链接日志一致 |

## 4. 性能解读（距预期 2.9 tok/s 的差距）

实测 2.1 vs 预期 2.9：CPU 收缩只覆盖 73.6% 字节，剩余 26% 权重的 GPU 段、
每 token 的 readback/sync 序列、以及 `rowsplit_gemm` 对 {34816,5120} 的
AVX2 路径（Phase 1c 为 {151936,5120} 的 output head 调优）共同占掉差值。
prefill 变慢（3.5→9.5s @ r=0.95）是 T>1 收缩线性放大的预期行为。

`rowsplit_bench`（直接 `#include` 生产源，短时交错 A/B）结论：生产 fp32 内核
gate_up {34816,5120} ≈155、down {5120,17408} ≈135 GFLOP/s；int8-split 变体
≈180-190 GFLOP/s（同会话交错 1.17~1.43×，rel_l2 1.9e-3 过 A16 判据 3.9e-3
但裕量仅 2×）。锁页修复后引擎内有效速率 ~120 GFLOP/s 已接近基准，**int8 的
边际收益不足 20%，性价比低于方向 1**，暂缓（实现与正确性对照保留在
`.workbuddy/tmp/rowsplit_bench.cpp`，随时可上）。

进一步提速方向（按性价比）：
1. `gdn_input_proj`/`attn_input_proj` 的 host 路由（18.8+5.2%）——它们的
   Q4/Q5 配对形态可复用 `linear_pair` independent 路线；**注意
   `linear_pair.cpp:96` 缺 k=4096（9B），启用前必须补**。
2. prefill 混合模式：prefill 走流送上传、decode 走 host 收缩（按 phase 切换），
   可把 prefill 9.5s 拉回 ~4s。
3. `rowsplit_gemm` int8-split 变体（见上）。

## 5. 复现

```powershell
# 构建
python .workbuddy/tmp/msvcbuild.py targets 89 ninfer
python .workbuddy/tmp/msvcbuild.py deploycrt 89

# smoke（三组对照 + 中文正确性）
powershell -File .workbuddy/tmp/hostlinear_smoke.ps1
# 结果: .workbuddy/tmp/hostlinear_result.txt
```

注意：ratio 越高 prefix 越小、设备占用越低；8GB 卡上 host-linear 建议
≥0.95，且 KV 需显式指定（本 smoke 用 `--kv-capacity 4096`，启动后剩余
540 MiB）。
