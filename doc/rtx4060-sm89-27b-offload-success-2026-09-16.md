# RTX 4060 sm_89：Qwen3.8-27B CPU offload 验收（2026-09-16 深夜）

继 9B P0（`rtx4060-sm89-offload-p0-success-2026-09-16.md`）之后，27B 端到端打通。

## 1. 权重获取

- 来源：**ModelScope** `Qwen/Qwen3.8-27B`（官方同权重，shard1 字节级一致），18 分片共 **51.75GB** → `D:\models\Qwen3.8-27B\`。
- hf-mirror 在一小时 18 路并发后对本 IP 限流（502/000），切换 ModelScope 后 **38.5MB/s**（3.5 倍）。
- 真实大小表从 `modelscope.cn/api/v1/models/<org>/<name>/repo/files?Recursive=true` 取回固化（`dl27b_sizes.json`），SKIP/完成/最终校验全走本地表。
- ★ 教训：`curl -sSLI` HEAD 预检的 Content-Length 不可靠（限流/错误页也带合法 CL，曾致 18 worker 全部假 SKIP）；完成判定必须 "curl exit 0 且字节数精确 == 真实大小"。

## 2. 转换

```
python -m tools.convert --model D:\models\Qwen3.8-27B --recipe qwen3_8_27b \
  --components text --device cpu --out D:\deps\models\qwen38-27b-final.ninfer --name qwen3_8_27b
```

- recipe `qwen3_8_27b` 已在 `official_recipes.py`（`_dense_groupwise(Q8)`），sm_89 走普通版（NVFP4 变体不可用）。
- torch 为 CPU 版 → `--device cpu`；**211.8s 完成**，产物 17.1GB（775 对象，q4/q5/q8_g64 混合 + bf16 norm）。
- chat_template：27B 官方模板**原样命中**引擎 `kReasoningEffortTemplateDigest`（chat_template.cpp:25），无需 9B 时代的 resource override；引擎原生支持 reasoning_effort 语义。

## 3. P0 实现两处关键修正（9B 时被掩盖）

1. **槽位改层内轮转**：原实现槽位 = `streamed_bytes`（全部流送字节），GPU 实际仍要装下全部权重——9B 显存恰好挤过去，27B 直接把 8GB 吃成 free=0（WDDM 超订阅 cudaMalloc 不报错）。修正：槽位 = **最大单层足迹**（27B ≈ 0.21GiB），`slot_offset` 按层内分配，`ensure_layer()` 覆盖写（`load.cpp` / `offload_stream.h`）。
2. **ratio 语义翻转**：原实现 ratio = "驻留 GPU 的比例"（0.7 → 前 70% 层驻留 → GPU 11.94GB）；按用户语义（"部分放到 CPU 上"）修正为 **ratio = 流送到 host 的比例**：`target_resident = (1-ratio) × layer_bytes`。`--offload-ratio 0.9` → GPU 驻留 3.99GiB、host 11.94GiB、first_streamed_layer=7。

另：27B 的 embedding+lm_head（bf16，≈3.1GB）不参与 offload，恒驻留——这决定了 ratio=0.7 时 GPU 仍需 6.7GB 在 8GB 卡上不可行，**27B 实际可用比例 ≥0.9**。

## 4. 验收（--offload-ratio 0.9，kv=ctx=4096）

| 指标 | 值 |
|---|---|
| 输出 | 「我是通义千问，一个由阿里巴巴云开发的AI助手。」中文正确 |
| GPU 权重 | **3.99 GiB**（产物 17.1GB 的 23%） |
| host 常驻 | 11.94 GiB（90% 层权重） |
| decode | **0.9 tok/s**（同步流送，与方案预估 ~1 tok/s 一致） |
| prefill | 4.3 tok/s |
| runtime reservation | 567.3 MiB（KV 256MB + workspace） |
| free after startup | 2.18 GiB（8GB 内从容） |
| CUDA Graph allowance | 0 B（强制关 ✓） |

命令行要点：`--kv-capacity` 单位是 **token 数**（非字节）且必须 `≤ max_concurrency × max_context`；KV auto 在 offload 下会因槽位+context 挤占而失败，**27B 必须显式 kv/ctx**。

## 5. 已知边界与 P1

- 槽位单缓冲同步流送 → decode 0.9 tok/s；P1 双缓冲 + transfer_stream 重叠是提速关键（预估 2-4 倍）。
- embedding/lm_head 常驻不可 offload；跨层共享对象在层轮转槽位下未支持（当前无此形态，出现会互踩，需要时改为对象常驻）。
- 与 speculative 互斥；9B 的 `--offload-ratio` 数值含义同步变化（0.7 现在 = 70% 上 CPU）。
