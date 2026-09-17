# Task F 交付：Qwen3.5-9B 在 RTX 4060 (sm_89) 真机冒烟成功

> 日期：2026-09-16 晚
> 产物：`D:\deps\models\qwen35-9b-final.ninfer`（391 objects，5.29 GB）

## 冒烟结果

```
engine ready | qwen3.5-9b | formats bf16,fp32,q4_g64_fp16,q5_g64_fp16,q6_g64_fp16
generate    text prefill                   84.5 ms     prefill speed   201.2 tok/s
generate    decode                          1.2s       decode speed     38.2 tok/s
summary     gpu weights used         5.28 GiB / 5.28 GiB
summary     planned device total     5.52 GiB
```

中文输出（`--messages` UTF-8 文件 + `--no-thinking`）：

> 我是 Qwen3.5，阿里巴巴最新推出的通义千问大语言模型，具备强大的逻辑推理、代码生成及多模态理解能力。

回归：12 套件 **11/12**（唯一 FAIL = 已知 27B T=4097 fp32 地板遗留），运行时修复零破坏。

## 本轮修复清单

### 数据侧
| 项 | 说明 |
|---|---|
| 权重下载 | hf-mirror 18.4GB 完成；分片 3 用 curl 直链 `-C -` 断点续传，`.incomplete` 手动改名归位 |
| generation_config.json | repo 未提供，手写（转换器必查但不解析） |
| chat_template | 引擎只认两个编译期 sha256 + tokenizer_config 内嵌模板逐字节一致。**已知 digest = 夹具 LF 归一且去末尾 \n**（=e84f32a2…）；生成 `D:\deps\tpl_engine.jinja` + `--resource` 覆盖 + 同串嵌入 tokenizer_config 后重转换 |

### 转换器 Windows 兼容（4 文件）
| 文件 | 修复 |
|---|---|
| `tools/artifact/file_io.py` | `os.sysconf`/`posix_fadvise`/`fdatasync` 降级；`pread`/`pwrite` 模拟（lseek 保存恢复 + 加锁） |
| `reader.py`×2、`safetensors.py` | `os.open` 加 **`O_BINARY`**（Windows 文本模式遇 0x1A 当 EOF → 假 short read） |
| `writer.py` | `os.pwrite` → 兼容版（mkstemp 本就二进制） |

### 运行时（4 处）
| 文件 | 修复 |
|---|---|
| `ops/weight_input.cpp:211` | GDN a/b 非连续配对回退只认 48 行(27B) → 加 32（9B 的 a/b 是两个独立对象） |
| `ops/wrapper/gdn_gating_proj.cpp` | 配对重载硬编码 48/5120 → `gdn_control_geometry()` 三档（norm 版同步） |
| `models/.../planning/startup.cpp:804` | 顶层架构门 `!=120` → 放行 89（op 层全修完也过不去，冒烟才暴露） |
| 环境文件 | torch/lib 遮蔽 System32 损坏 CRT（c10.dll WinError 1114） |

## 经验教训（详见 .workbuddy/memory）

1. `os.open` 在 Windows 不带 `O_BINARY` = 文本模式，读权重必坏。
2. `main(argv)` 按 ANSI(GBK) 收参 → 中文必须走 `--messages` UTF-8 文件。
3. safe-delete 钩子拦截/回滚全系统删除 → 产物删除不可靠，用新文件名输出。
4. bash 沙箱视图可与真实 FS 脱同步 → 文件真伪用 Read/Glob 交叉验证。

## 下一步

- 27B offload（方案已出：`rtx4060-qwen38-27b-cpu-offload-plan.md`，先在 9B 上以强制 30% 常驻验证 P0/P1）
- KV 容量拉长实测（auto 只给 2048，可显式传 --kv-capacity）
- `linear_pair.cpp:96` 补 k=4096（启用 dflash/proposal 前）
