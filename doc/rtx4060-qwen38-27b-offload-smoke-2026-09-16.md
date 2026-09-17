# Qwen3.8-27B CPU Offload @ RTX 4060 (sm_89) 冒烟报告 — 2026-09-16

## 结论

**冒烟通过。** 27B 产物 `D:\deps\models\qwen38-27b-final.ninfer`（17.1 GB，Q6 系 packed，
recipe `qwen3_8_27b`，963 参数 / 609 uses / 775 objects）在 RTX 4060 Laptop 8GB 上以
CPU offload 模式完成端到端生成：正常输出中文 token、逐字无异常、进程零崩溃。

## 验收数据（--offload-ratio 0.8）

| 项目 | 数值 |
|---|---|
| GPU 权重驻留 | **5.24 GiB**（总权重 15.93 GiB，host 驻留 10.69 GiB） |
| 权重加载 | 8.5s @ 629 MiB/s（h2d 5.24 GiB） |
| offload 槽位 | 0.21 GiB，首层 = 13 |
| prefill | 5.6 tok/s（17 prompt tokens, 3.1s） |
| decode | **1.0 tok/s**（29 tokens, 29.3s） |
| KV cache | bf16, capacity 2048（auto 策略）, payload 128 MiB |
| 显存余量 | 权重后 free 1.48 GiB；启动后 free 1.05 GiB |
| 采样 | greedy, no-thinking, finish reason = stop-token |
| 输出 | 中文正常："我是通义千问，一个由阿里巴巴通义实验室独立开发……" |

## 关键发现：offload-ratio 0.7 在 8GB 卡上不可行（根因/修复/验证）

- **根因**：ratio 0.7 时 GPU 驻留权重 6.70 GiB（offload 比例按对象数近似传递，字节占比
  ≠ 1-ratio）+ 槽位 0.21 GiB 后，8GB 显存仅剩 23 MB；`--kv-capacity auto` 强制要求
  1 GiB 余量 → startup 直接 fail（非崩溃，报错干净退出）。
- **修复**：提高 offload-ratio 至 0.8。GPU 驻留降至 5.24 GiB，KV headroom 1.00 GiB 满足。
- **验证**：`smoke27b_run2.log`，EXIT=0，29 token 全部生成。
- **注意**：此前备忘"槽位显存不计入 kv auto"在 27B 上表现为更严格——不仅槽位，字节级
  offload 粒度也使 `(1-ratio)*总权重` 与实际驻留偏差显著，8GB 卡排 27B 应从 0.8 起步。

## 已知边界与下一步（P1/P2 范畴，本轮不做）

- decode 1.0 tok/s 为 CPU PCIe 流送的真实水位；MoE hint / 双缓冲（P1/P2）是提速方向。
- CUDA Graph allowance 0 B（offload 与 graph 互斥，符合 P0 设计）。
- 槽位显存不计入 kv auto：27B 实际部署建议显式 `--kv-capacity`。

## 复现命令

```
ninfer.exe D:\deps\models\qwen38-27b-final.ninfer ^
  --messages <UTF-8 JSON> --max-new 48 --greedy --no-thinking ^
  --kv-capacity auto --offload-ratio 0.8
```
（PATH 需含 D:\deps\ffmpeg\bin、D:\deps\curl\bin、CUDA v13.2 bin；中文提示必须走 --messages 文件）

原始日志：`.workbuddy/tmp/smoke27b_run2.log`（成功）/ `smoke27b_run1.log`（0.7 失败样本）
