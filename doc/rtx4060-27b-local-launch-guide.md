# RTX 4060 本地启动指南（Qwen3.8-27B / 9B）

日期：2026-09-17
适用：sm_89 构建（RTX 4060 Laptop 8GB）、Windows + PowerShell 5.1
相关文档：`rtx4060-27b-offload-speedup-plan-2026-09-16.md`（提速方案与基准）、`rtx4060-sm89-support-solution.md`（移植方案）

---

## 1. 这个脚本解决什么问题

27B 在 8GB 卡上不是一个「直接跑」就能跑起来的组合。裸调 `ninfer.exe` 需要同时摆平六件事，任一遗漏都表现为「跑不起来」或「结果乱码」：

| # | 问题 | 缺了会怎样 | 脚本里的对应处理 |
|---|---|---|---|
| 1 | 权重装不下显存 | 直接 OOM / 加载失败 | 默认 `--offload-ratio 0.9`（27B 必须 ≥ 0.8） |
| 2 | offload 下 `--kv-capacity auto` 必然失败 | 启动期报错 | 显式下发 `--kv-capacity` 与 `--max-context` |
| 3 | 投机解码提速 2.2× 但默认关闭 | 慢一倍 | 默认 `--spec mtp --draft-tokens 3` |
| 4 | `main()` 按 GBK 接收 argv | 中文提示词变乱码 | 提问统一写入**无 BOM 的 UTF-8 JSON** 再传 `--messages` |
| 5 | 可执行文件依赖 ffmpeg / libcurl 的 DLL | 起不来（找不到 DLL） | 启动前把 `D:\deps\ffmpeg\bin`、`D:\deps\curl\bin` 塞进 PATH |
| 6 | System32 里的 CRT 是 2016 年的旧版 | 第一个 mutex 加锁就 `0xC0000005` | 自动检测并在可执行文件旁部署 VC143 CRT |
| 7 | serve 端默认要 pin 8 GiB 主机 KV 做上下文缓存 | 与 offload 的 ~12 GiB 主机权重叠加后 `cudaMallocHost` OOM | offload 打开时自动改为 `--host-kv-mib 0` |

再加一条界面层的：模型输出是 UTF-8 字节流，控制台不切到 UTF-8 会花屏 —— 脚本在入口就把控制台输入输出编码切成 UTF-8。

---

## 2. 前置条件

| 项目 | 要求 |
|---|---|
| 构建产物 | `D:\deps\build\ninfer-89\apps\ninfer.exe`、`ninfer-serve.exe`（sm_89 构建） |
| 模型产物 | `D:\deps\models\qwen38-27b-q4v3mtp.ninfer`（推荐，15.45 GiB） |
| 依赖 DLL | `D:\deps\ffmpeg\bin`、`D:\deps\curl\bin` |
| 显卡 | RTX 4060 Laptop 8GB（`doctor` 会打印实测显存） |
| 首次加载耗时 | 27B 约 30~60 秒（读取 15.45 GiB 权重并页锁 host 侧副本） |

先跑一次体检确认全部就绪：

```bat
run-27b.bat doctor
```

`doctor` 会逐项打印可执行文件时间戳、CRT 状态、依赖目录、GPU、可用模型产物与推荐命令 —— 任何一项不是 `[  ok  ]`，就该先处理它。

---

## 3. 快速开始

```bat
REM 1) 体检
run-27b.bat doctor

REM 2) 单轮提问
run-27b.bat chat -Prompt "用一句话介绍你自己。"

REM 3) 打开网页聊天（推荐：中文输入与流式输出都没有终端编码烦恼）
run-27b.bat web

REM 4) 起 API 服务给别的客户端用
run-27b.bat serve -Port 8000
```

双击 `run-27b.bat`（不带参数）等价于 `doctor`，窗口会自动停留以便阅读。

---

## 4. 模式一览

| 模式 | 用途 | 权重加载次数 | 典型命令 |
|---|---|---|---|
| `doctor` | 环境体检（默认） | 不加载 | `run-27b.bat doctor` |
| `chat` | 单轮生成，结果打到终端 | 每次 1 次 | `run-27b.bat chat -Prompt "..."` |
| `repl` | 终端多轮对话 | 仅 1 次 | `run-27b.bat repl` |
| `serve` | 前台运行 API 服务 | 仅 1 次 | `run-27b.bat serve -Port 8000` |
| `web` | 起服务 + 打开网页聊天页 | 仅 1 次 | `run-27b.bat web` |
| `bench` | 吞吐基准（decode / prefill / 接受率） | 1 次 | `run-27b.bat bench` |
| `stop` | 停掉后台残留的 `ninfer-serve` | — | `run-27b.bat stop` |

**为什么推荐 `web` / `repl` 而不是反复 `chat`**：27B 每次重启都要重新读取并页锁 15.45 GiB 权重，约 30~60 秒。多轮场景一律用常驻服务。

---

## 5. 参数速查

### 5.1 常用

| 参数 | 默认 | 说明 |
|---|---|---|
| `-Model <路径>` | 自动挑选 | 省略时优先 `qwen38-27b-q4v3mtp.ninfer` |
| `-ModelDir <目录>` | `D:\deps\models` | 自动挑选时的搜索目录 |
| `-AppDir <目录>` | `D:\deps\build\ninfer-89\apps` | 可执行文件所在目录 |
| `-OffloadRatio <F>` | `0.9` | 流送到主机内存的文本层比例；27B ≥ 0.8，9B 自动取 0 |
| `-DraftTokens <N>` | `3` | MTP 草稿窗口，实测 3 是甜点 |
| `-MaxNew <N>` | `512` | 单次最多生成 token 数 |
| `-MaxContext <N>` | `4096` | 上下文上限（token） |
| `-KvCapacity <N>` | 同 MaxContext | KV 容量（token），必须 ≥ MaxContext |
| `-Thinking` | 关 | 打开思维链（默认 `--no-thinking`） |
| `-Greedy` | 关 | 强制贪心解码（temperature 0） |
| `-Temperature <F>` | 模型默认 | 不传则不覆盖模型自带采样参数 |
| `-FixCrt` | 关 | 强制重新部署 VC143 CRT（**每次重建后建议加**） |
| `-ExtraArgs a b` | 空 | 透传任意额外 CLI 参数 |

### 5.2 仅 serve / repl / web

| 参数 | 默认 | 说明 |
|---|---|---|
| `-Port <N>` | `8080` | 监听端口 |
| `-BindHost <IP>` | `127.0.0.1` | 改 `0.0.0.0` 才对局域网开放（注意无鉴权风险） |
| `-ApiKey <串>` | 空 | 非空则要求 `Authorization: Bearer <串>` |
| `-MaxConcurrency <N>` | `1` | 并发序列数；KV 容量需 ≤ `MaxConcurrency × MaxContext` |
| `-HostKvMiB <N>` | `-1`（自动） | 跨请求上下文缓存的主机 KV（MiB）。offload 打开时自动取 `0`，否则不传 |
| `-Detach` | 关 | `web` 模式起好服务就退出，服务留后台（用 `stop` 停） |
| `-NoOpen` | 关 | `web` 模式不自动打开浏览器，只打印 URL |

> `repl` / `web` 起的后台服务日志被压到 `warning` 并关掉周期统计（`--log-stats-interval-ms 0`），
> 避免服务日志打断对话流式显示。要看完整启动日志请用 `serve` 模式前台跑。

### 5.3 其它

| 参数 | 说明 |
|---|---|
| `-PromptFile <UTF-8 文件>` | 从文件读提问（含引号/反斜杠的提问建议走这条） |
| `-Log <文件>` | 把输出同时落盘；注意经管道后实时流式会退化成按行刷新 |
| `-KvDtype <bf16\|int8\|fp8>` | KV 缓存量化格式（默认 bf16；sm_89 不支持 nvfp4/k8v4） |
| `-Device <N>` | 多卡时选设备，默认 0 |
| `-NoSpec` | 关闭 MTP，退回逐 token 解码（用于对照） |

---

## 6. 各模式用法

### 6.1 `chat` —— 单轮生成

```bat
run-27b.bat chat -Prompt "用一句话介绍你自己。"
run-27b.bat chat -PromptFile D:\deps\q.txt -MaxNew 256 -Log D:\deps\out.log
run-27b.bat chat -Prompt "..." -Thinking          REM 打开思维链
run-27b.bat chat -Prompt "..." -Greedy            REM 确定性输出
run-27b.bat chat -Model D:\deps\models\qwen35-9b-final.ninfer -Prompt "你好"
```

内容走 stdout、诊断与统计走 stderr，因此不改动的情况下终端是「先看到答案、后看到统计」。

### 6.2 `repl` —— 终端多轮对话

```bat
run-27b.bat repl
```

脚本自己在后台起一个 `ninfer-serve`，等 `/health` 就绪后进入对话循环，退出时自动收尾。支持三个命令：

| 命令 | 作用 |
|---|---|
| `:exit` / `:q` | 退出（同时停掉后台服务） |
| `:reset` | 清空上下文 |
| `:file <路径>` | 从 UTF-8 文本文件读入提问（**中文输入异常时的兜底**） |

> 中文输入依赖控制台代码页。脚本已把输入输出都切到 UTF-8；个别终端（旧版 conhost、部分 SSH 客户端）仍可能吞中文，此时把问题存成 UTF-8 的 `.txt` 用 `:file`，或直接用 `web` 模式。

### 6.3 `serve` —— API 服务（前台）

```bat
run-27b.bat serve
run-27b.bat serve -Port 8000 -MaxContext 8192 -KvCapacity 8192
run-27b.bat serve -MaxConcurrency 4 -KvCapacity 16384 -MaxContext 4096
run-27b.bat serve -ApiKey sk-local -BindHost 0.0.0.0 -NoOpen
```

`ninfer-serve` 同时提供三套协议：

| 协议 | 端点 |
|---|---|
| OpenAI Chat Completions | `POST /v1/chat/completions`（支持 `stream`） |
| OpenAI Responses | `POST /v1/responses` 系列 |
| Anthropic Messages | `POST /v1/messages`、`POST /v1/messages/count_tokens` |
| 运维 | `GET /health`、`GET /v1/models` |

### 6.4 `web` —— 网页聊天（最省事）

```bat
run-27b.bat web
run-27b.bat web -Detach        REM 起好就退出，服务留后台
run-27b.bat stop               REM 用完停掉
```

流程：后台起 `ninfer-serve --cors`（日志压到 `warning`、关闭周期统计，避免打断页面之外的终端输出）
→ 轮询 `/health` 直到就绪（进度以 `...` 显示）→ 用默认浏览器打开 `tools/local_chat/index.html?port=<端口>`。

- 默认（不带 `-Detach`）脚本会一直守着服务，**关闭窗口即停止服务**。
- 带 `-Detach` 时脚本立即退出、服务在独立控制台里继续跑，**关窗口也不会停**，用 `run-27b.bat stop` 收掉。

页面本身是纯静态单文件（无外部依赖），通过 URL 上的 `port` / `host` / `key` 连接本地服务，功能：流式输出、思维链折叠显示（灰色小字）、`enable_thinking` 开关、最大长度选择、停止生成、清空上下文、实时「字数 / 秒 / 字每秒」统计。

`file://` 页面跨源访问本地端口依赖服务的 `--cors`（脚本已自动加），`Access-Control-Allow-Origin: *` 允许该场景。

### 6.5 `bench` —— 吞吐基准

```bat
run-27b.bat bench
run-27b.bat bench -DraftTokens 2
run-27b.bat bench -NoSpec              REM 关投机做对照
run-27b.bat bench -Model D:\deps\models\qwen38-27b-q4v2.ninfer -MaxNew 256
```

固定用同一条长回复 prompt + `--greedy`，跑完打印这张表：

| 指标 | 含义 |
|---|---|
| `prompt tokens` / `generated tokens` | 输入 / 输出 token 数 |
| `model elapsed` | 纯模型耗时（不含加载） |
| `prefill speed` / `decode speed` | 两阶段吞吐 |
| `mtp acceptance rate` | 草稿接受率（衡量 MTP 是否划算） |
| `mtp acceptance length` | 每轮流送平均提交多少 token（**这个数就是投机解码的实际倍率**） |
| `gpu weights used` | 显存权重占用 |

完整日志落在 `%TEMP%\ninfer-launch\bench.log`。

### 6.6 `stop`

```bat
run-27b.bat stop
```

停掉所有残留的 `ninfer-serve` 进程（`web -Detach` 之后必用，否则占着显存）。

---

## 7. 客户端接入示例

### 7.1 curl

```bat
curl http://127.0.0.1:8080/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"max_tokens\":128}"
```

带鉴权时加 `-H "Authorization: Bearer sk-local"`。

### 7.2 Python（OpenAI SDK）

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

stream = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "用一句话介绍你自己。"}],
    max_tokens=256,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

模型名以 `GET /v1/models` 返回的 `data[0].id` 为准（多数客户端填 `local` 也能过）。

### 7.3 Anthropic Messages

```bat
curl http://127.0.0.1:8080/v1/messages ^
  -H "Content-Type: application/json" ^
  -H "anthropic-version: 2023-06-01" ^
  -d "{\"model\":\"local\",\"max_tokens\":256,\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}"
```

### 7.4 第三方 GUI

任何支持自定义 OpenAI 端点的客户端（Cherry Studio、Open WebUI、Continue、各类 IDE 插件）都能直接接：Base URL `http://127.0.0.1:8080/v1`，API Key 留空或填 `-ApiKey` 指定的值。

---

## 8. 参数怎么选

### 8.1 `-OffloadRatio`

| 模型 | 建议 | 原因 |
|---|---|---|
| Qwen3.8-27B | `0.9`（**不得低于 0.8**） | embedding + lm_head 约 3.1 GiB 恒驻留且不可 offload，8GB 卡再低就装不下 |
| Qwen3.5-9B | `0`（脚本自动） | 5.29 GiB 权重直接装得下，offload 反而慢 |

比例越高越省显存、越慢（每 token 要走一遍 PCIe）。0.9 时 GPU 权重 4.31 GiB、剩余显存供 KV 用。

### 8.2 `-DraftTokens`

实测（同 prompt、148 token、ratio 0.9、greedy）：

| k | decode | 相对基线 | 说明 |
|---|---|---|---|
| 关投机 | 0.9 tok/s | 1.00× | — |
| 2 | 1.8 tok/s | 2.00× | |
| **3** | **2.0 tok/s** | **2.22×** | **默认，甜点** |
| 5 | 2.0 tok/s | 2.22× | 无进一步收益，草稿越深接受率越低 |

`tok/round ≈ 2.2`（即接受率约 60%）说明每轮流送平均提交 2.2 个 token。任务越结构化（代码、摘要、改写），接受率越高，收益越大；自由创作型反而偏低。

### 8.3 `-MaxContext` / `-KvCapacity`

单位都是 **token**（不是字节），且必须满足 `KvCapacity ≥ MaxContext`（脚本会自动纠正）。多并发时还要满足 `KvCapacity ≤ MaxConcurrency × MaxContext`。

```bat
REM 单序列 4K 上下文
run-27b.bat serve -MaxContext 4096 -KvCapacity 4096

REM 4 条序列共享 16K KV
run-27b.bat serve -MaxConcurrency 4 -MaxContext 4096 -KvCapacity 16384
```

显存紧张就调小 KV：`-MaxContext 2048 -KvCapacity 2048`。

### 8.4 `-HostKvMiB`

只影响 `serve` / `repl` / `web`。serve 端默认要 pin 8 GiB 主机内存做跨请求上下文缓存；offload 已经把约 12 GiB 权重压在主机侧（同样是 pin 内存），两者叠加实测直接
`ERROR startup failed | pinning host KV` → `cudaMallocHost failed: cudaErrorMemoryAllocation`。
所以脚本在 offload 打开时默认传 `--host-kv-mib 0`（本机 31.7 GB 内存也顶不住 12 + 8 + 1.15 GiB 的锁定页）。
机器内存充裕、又需要跨请求前缀复用收益时，可以显式给一个值，例如 `-HostKvMiB 2048`。

---

## 9. 故障排查

| 症状 | 根因 | 处理 |
|---|---|---|
| 启动即 `0xC0000005`（访问冲突，第一次加锁时） | System32 的 `msvcp140/vcruntime140` 是 2016 年旧版 | `run-27b.bat doctor -FixCrt`（重建后必做） |
| 进程起不来、提示找不到 DLL | ffmpeg / libcurl 的 bin 不在 PATH | 确认 `D:\deps\ffmpeg\bin`、`D:\deps\curl\bin` 存在；脚本会自动加 |
| `unknown argument: --offload-ratio` | `ninfer-serve.exe` 是旧构建，或 serve 解析器缺该分支 | 重建 `ninfer-serve`（见 §10）；解析分支已在 `serve_options.cpp` 补齐 |
| `pinning host KV` → `cudaMallocHost failed: out of memory` | serve 端 8 GiB 主机 KV 与 offload 的主机权重叠加超限 | `-HostKvMiB 0`（offload 场景脚本已默认如此） |
| `unsupported single-parent format` 之类加载报错 | 模型产物的量化格式与算子表不匹配 | 换回已验证的产物（如 `q4v3mtp` / `q4v2`），或补齐对应形状 |
| 中文输出花屏 | 控制台不是 UTF-8 | 用 `run-27b.bat` 启动（脚本会切编码）；仍不行就用 `web` |
| 中文提示词变问号/乱码 | argv 按 GBK 解析 | 用 `-PromptFile` 或 `:file`；`chat -Prompt` 已由脚本转 UTF-8 JSON |
| 自己把输出重定向/管道捕获后，中文变成 `鍩哄噯` 式乱码 | 父进程按 GBK 解码了子进程的 UTF-8 字节流 | 终端里直接运行是正常的；要捕获就在父会话先 `[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)`，或直接读 `%TEMP%\ninfer-launch\*.log`（子进程直写，不经这层转换） |
| 双击 `run-27b.bat` 闪退 / 参数未生效 | `.bat` 的换行符必须是 CRLF，且只含 ASCII | 已固定为 CRLF + 纯 ASCII；改动时不要引入 LF 或中文 |
| 加载报显存不足 | ratio 太低或 KV 太大 | 提高 `-OffloadRatio` 到 0.9，或调小 `-MaxContext`/`-KvCapacity` |
| 比预期慢一倍 | 没开投机解码 | 确认用的是含 MTP 的产物（名字带 `mtp`），不要加 `-NoSpec` |
| 退出后显存没释放 | 残留的后台服务 | `run-27b.bat stop` |
| 每轮都要等 30~60 秒 | 反复 `chat` 重启进程 | 改用 `repl` / `serve` / `web`，权重只加载一次 |

---

## 10. 与其它工具的关系

| 工具 | 职责 |
|---|---|
| `build-sm89.bat` | 从零配置并构建 sm_89 的构建树 |
| `.workbuddy/tmp/msvcbuild.py` | 本机受限 shell 下的构建驱动（`targets` / `deploycrt`） |
| `.workbuddy/tmp/run_tests.py` | 12 套件回归测试 |
| **`run-27b.bat` / `run-27b.ps1`** | **运行期启动器**（本指南） |
| `tools/local_chat/index.html` | 网页聊天页，供 `web` 模式加载 |

**构建 → 运行的标准顺序**：

```bat
REM 1) 重建（改过代码后）
.workbuddy\tmp\msvcbuild.py targets 89 ninfer ninfer-serve

REM 2) 重新部署 CRT（每次重建后必做）
.workbuddy\tmp\msvcbuild.py deploycrt 89

REM 3) 体检 + 启动
run-27b.bat doctor
run-27b.bat web
```

或者把第 2 步交给启动器：`run-27b.bat web -FixCrt`。

### 维护须知（改脚本前必读）

| 事项 | 说明 |
|---|---|
| **`ninfer-serve` 是独立目标** | 引擎库一改就得单独重建它，否则 `serve`/`repl`/`web` 会用旧二进制（曾因此报 `unknown argument: --offload-ratio`） |
| **`.ps1` 必须带 UTF-8 BOM** | PowerShell 5.1 对无 BOM 的 UTF-8 文件按 ANSI 解析，中文会吞掉换行符并引发语法错误。`run-27b.ps1` 已带 BOM，重写后请确认 |
| **不要用 `Start-Process` 起服务** | PS 5.1 会重建子进程环境字典，进程环境块里只要有仅大小写不同的重复变量（`Path`/`PATH`、`http_proxy`/`HTTP_PROXY`）就抛「已添加项」。脚本用 `System.Diagnostics.Process` 让子进程直接继承环境块 |
| **`$PSBoundParameters` 有作用域陷阱** | 它在函数内部看到的是函数自己的绑定集合。脚本在顶层抄了 `$script:TemperatureSpecified` / `$script:OffloadRatioSpecified` 供函数使用 |
| **`$ErrorActionPreference='Stop'` + 原生 stderr** | 把原生命令的 stderr 并进管道会被当成终止性错误而中断脚本；`Invoke-Native` 局部放宽并把 ErrorRecord 还原成原始文本行 |

---

## 11. 新增文件

| 文件 | 说明 |
|---|---|
| `run-27b.ps1` | 启动器主体（UTF-8 with BOM，PowerShell 5.1 才能正确读取中文） |
| `run-27b.bat` | cmd / 双击入口（纯 ASCII，仅转发参数并绕过执行策略） |
| `tools/local_chat/index.html` | 自包含网页聊天页（无外部依赖） |
| `doc/rtx4060-27b-local-launch-guide.md` | 本文件 |

代码侧另有一处修复：`src/serve/serve_options.cpp` 补上了缺失的 `--offload-ratio` 解析分支
（此前 `ServeOptions` 有字段、用法文本有说明、`generation_service` 有透传，唯独解析器不认，
导致 `ninfer-serve` 报 `unknown argument`）。

---

## 12. 验证记录（2026-09-17）

模型 `qwen38-27b-q4v3mtp.ninfer`，`--offload-ratio 0.9 --spec mtp --draft-tokens 3`，RTX 4060 Laptop 8GB。

| 模式 | 验证内容 | 结果 |
|---|---|---|
| `doctor` | 可执行文件 / CRT / 依赖目录 / GPU / 模型清单 | ✅ 全部 `[  ok  ]`，GPU 8188 MiB |
| `chat` | 中文提问 → 中文输出 + 统计 | ✅ exit 0，decode **2.5 tok/s**，接受率 **66.7%**，3.00 tok/round |
| `web -Detach` | 后台起服务 → `/health` → `/v1/models` | ✅ `{"status":"ok"}`、模型 id `qwen3_8_27b` |
| 同上 | HTTP 非流式补全 | ✅ 17.5 s / 24 token，中文正确 |
| 同上 | HTTP 流式（SSE） | ✅ `data:` 帧 + `[DONE]`，首字 4.7 s |
| `stop` | 停掉后台服务 | ✅ 端口释放、无残留进程 |
| `repl` | 多轮：中文输入 → 流式回答 → `:exit` | ✅ exit 0，回答中文正确 |
| `serve` | 前台起服务 → `/health` / `/v1/models` | ✅ 均正常 |
| `bench` | 48 token 基准 → 指标表 | ✅ exit 0，decode **1.8~1.9 tok/s**、接受率 **35.3%**、**2.04 tok/round**、GPU 权重 4.31 GiB |

> `chat` 与 `bench` 的接受率差异（66.7% vs 35.3%）来自提示词：`bench` 用的是自由创作型长文本（十二句写景），
> 草稿命中率天然更低；这也是 §8.2 里「任务越结构化收益越大」的实测证据。

**过程中定位并修掉的三个真实缺陷**（都不是脚本笔误，而是会真实咬人的坑）：

| # | 现象 | 根因 | 处理 |
|---|---|---|---|
| 1 | `chat` 一启动就中断，exit 1 | `$ErrorActionPreference='Stop'` 下把原生命令 stderr 并进管道 → `NativeCommandError` 被当成终止性错误 | 新增 `Invoke-Native`，局部放宽并把 ErrorRecord 还原成原始文本行（日志也因此不再出现 `CategoryInfo` 噪声） |
| 2 | `web`/`serve` 起不来：`unknown argument: --offload-ratio` | `ninfer-serve.exe` 是 9-16 的旧构建；且 serve 解析器本身缺该分支 | 重建 `ninfer-serve` + 补齐 `serve_options.cpp` 解析分支 |
| 3 | 服务活 2 秒后 `cudaMallocHost failed: out of memory` | serve 默认 pin 8 GiB 主机 KV 与 offload 的 ~12 GiB 主机权重叠加超限 | offload 打开时自动下发 `--host-kv-mib 0` |

**收尾回查（2026-09-17 00:45）**——补跑 `bench` 时又抓到两处小问题：

| # | 现象 | 根因 | 处理 |
|---|---|---|---|
| 4 | 指标表里 `mtp tok/round` 恒显示 `-` | 引擎 `summary` 里这一行的名字是 `mtp acceptance length`，脚本按错误的标签去匹配，永远命中不了（`bench` 之外看不出来） | 标签改为 `mtp acceptance length`，8 个指标现已全部解析成功 |
| 5 | `run-27b.bat` 是 LF 换行 | `cmd.exe` 解析含 `if (...) else (...)` 的块时 CRLF 才稳妥 | 固定为 CRLF + 纯 ASCII（`.ps1` 保持 UTF-8 with BOM + LF） |
