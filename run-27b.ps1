#Requires -Version 5.1
<#
.SYNOPSIS
    NInfer 本地启动器 —— Qwen3.8-27B @ RTX 4060 Laptop 8GB（权重流送 offload + MTP 投机解码）。

.DESCRIPTION
    把 27B 在 8GB 卡上跑起来所需的全部非默认设置收进一条命令：

      * 权重流送 offload    --offload-ratio 0.9（27B 必须 >= 0.8，见下）
      * MTP 投机解码        --spec mtp --draft-tokens 3（实测最甜点，k=5 无增益）
      * 显式 KV 容量        offload 下 --kv-capacity auto 必然失败，必须显式给
      * 中文提示词          main() 按 GBK 接收 argv，直接传中文会乱码 -> 统一转 UTF-8 JSON
      * ffmpeg/libcurl DLL  可执行文件依赖它们，必须进 PATH
      * VC143 CRT 旁加载    System32 那套是 2016 年的旧版，会 0xC0000005
      * 控制台 UTF-8        模型输出是 UTF-8 字节流，不改代码页会花屏

    模式（-Mode）：
      doctor   环境体检（默认）：可执行文件 / CRT / 模型产物 / 显存 / 推荐参数
      chat     单轮生成，结果直接打到终端
      repl     起本地服务后进入多轮对话（权重只加载一次，避免每轮重新流送）
      serve    前台运行 OpenAI + Anthropic 兼容 API 服务
      web      起服务并打开内置网页聊天页（推荐：中文输入与流式输出都没有终端编码烦恼）
      bench    长文本吞吐基准，打印 decode / prefill / 接受率
      stop     停掉后台残留的 ninfer-serve 进程

.PARAMETER Mode
    见上。默认 doctor。

.PARAMETER Prompt
    chat 模式的提问文本。中文可直接传（脚本负责转 UTF-8 消息文件）。
    含 shell 特殊字符时建议改用 -PromptFile。

.PARAMETER PromptFile
    从 UTF-8 文本文件读取提问（每行会被拼接）。repl 模式下也可用 :file <路径>。

.PARAMETER Model
    模型产物路径。省略时按下述优先级在 -ModelDir 中自动挑选：
    qwen38-27b-q4v3mtp.ninfer > qwen38-27b-q4v2.ninfer > qwen38-27b-final.ninfer > qwen35-9b-final.ninfer

.PARAMETER OffloadRatio
    流送到主机内存的文本层权重比例。0 = 全部驻留显存（仅小模型可行）。
    27B 在 8GB 卡上必须 >= 0.8（embedding + lm_head 约 3.1GB 恒驻留、不可 offload）。

.PARAMETER DraftTokens
    MTP 草稿窗口。默认 3（实测 2 -> 2.00x，3 -> 2.22x，5 -> 2.22x 无进一步收益）。

.PARAMETER MaxContext / KvCapacity
    上下文上限与 KV 容量（单位都是 token，且要求 KvCapacity >= MaxContext）。
    显式给这两个值是 offload 场景的硬要求。

.PARAMETER MaxConcurrency
    仅 serve/repl/web：并发序列数。KV 容量需 <= MaxConcurrency * MaxContext。

.PARAMETER HostKvMiB
    仅 serve/repl/web：跨请求上下文缓存的主机 KV 容量（MiB）。
    默认 -1 = 自动：offload 打开时取 0（offload 已占主机内存，再 pin 8 GiB 必然 OOM），
    offload 关闭时不传该参数、沿用服务端默认。

.PARAMETER Temperature
    采样温度。不传则使用模型自带默认；传了 -Greedy 时忽略。
    两者都不传时保持模型默认采样（不显式覆盖）。

.PARAMETER Thinking
    打开思维链。默认关闭（--no-thinking），27B 不开思考时响应更快、更省 token。

.PARAMETER NoSpec
    关闭 MTP 投机解码，退回纯逐 token 解码（用于对照与排查）。

.PARAMETER FixCrt
    重新把 VC143 CRT 复制到可执行文件旁边。每次重新构建之后都要做一次。

.PARAMETER Log
    把输出同时写到该文件（chat / bench 模式）。注意：经过管道后实时流式会变成按行输出。

.PARAMETER Detach
    web 模式：起好服务、打开浏览器后脚本就退出，服务留在后台继续跑（用 -Mode stop 停止）。
    后台服务日志被压到 warning，且运行在独立控制台里 —— 要看完整启动日志请用 -Mode serve。

.EXAMPLE
    .\run-27b.bat doctor
.EXAMPLE
    .\run-27b.bat chat -Prompt "用一句话介绍你自己。"
.EXAMPLE
    .\run-27b.bat web
.EXAMPLE
    .\run-27b.bat serve -Port 8000 -MaxContext 8192 -KvCapacity 8192 -FixCrt
.EXAMPLE
    .\run-27b.bat bench -DraftTokens 3 -MaxNew 256
.EXAMPLE
    .\run-27b.bat stop

.NOTES
    完整使用说明：doc/rtx4060-27b-local-launch-guide.md
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('doctor', 'chat', 'repl', 'serve', 'web', 'bench', 'stop')]
    [string]$Mode = 'doctor',

    [Parameter(Position = 1)]
    [string]$Prompt,

    [string]$PromptFile,
    [string]$Model,
    [string]$ModelDir = 'D:\deps\models',
    [string]$AppDir   = 'D:\deps\build\ninfer-89\apps',

    [double]$OffloadRatio = 0.9,
    [int]$DraftTokens     = 3,
    [int]$MaxNew          = 512,
    [int]$MaxContext      = 4096,
    [int]$KvCapacity      = 4096,
    [int]$MaxConcurrency  = 1,
    [int]$Device          = 0,
    [string]$KvDtype      = '',
    [string]$BindHost = '127.0.0.1',
    [int]$Port        = 8080,
    [string]$ApiKey   = '',
    [int]$HostKvMiB   = -1,

    [double]$Temperature = -1,
    [switch]$Greedy,
    [switch]$Thinking,
    [switch]$NoSpec,

    [string]$LogLevel = 'info',
    [string]$Log      = '',
    [switch]$FixCrt,
    [switch]$Detach,
    [switch]$NoOpen,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'

# $PSBoundParameters 只在脚本作用域里反映用户真正传了哪些参数；函数内部看到的是自己的绑定集合，
# 所以在这里先抄一份，供下面的函数判断「用户是否显式指定过」。
$script:TemperatureSpecified  = $PSBoundParameters.ContainsKey('Temperature')
$script:OffloadRatioSpecified = $PSBoundParameters.ContainsKey('OffloadRatio')

# ---------------------------------------------------------------------------
# 0. 控制台编码：模型输出按 UTF-8 字节写出，输入也要按 UTF-8 读，否则中文两边都花屏。
# ---------------------------------------------------------------------------
try {
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [Console]::OutputEncoding = $utf8
    [Console]::InputEncoding  = $utf8
    $OutputEncoding           = $utf8
} catch {
    # 某些宿主不允许改代码页；不影响 ASCII 场景。
}

$script:RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$script:ExeCli   = Join-Path $AppDir 'ninfer.exe'
$script:ExeServe = Join-Path $AppDir 'ninfer-serve.exe'
$script:TmpDir   = Join-Path $env:TEMP 'ninfer-launch'

function Write-Say  { param([string]$Text) Write-Host "[ninfer] $Text" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Text) Write-Host "[  ok  ] $Text" -ForegroundColor Green }
function Write-Note { param([string]$Text) Write-Host "[ note ] $Text" -ForegroundColor DarkGray }
function Write-Warn { param([string]$Text) Write-Host "[ warn ] $Text" -ForegroundColor Yellow }
function Write-Bad  { param([string]$Text) Write-Host "[ fail ] $Text" -ForegroundColor Red }

function Get-Invariant([double]$Value) {
    return $Value.ToString([System.Globalization.CultureInfo]::InvariantCulture)
}

# ---------------------------------------------------------------------------
# 1. 环境准备
# ---------------------------------------------------------------------------
function Initialize-Environment {
    # ffmpeg / libcurl 的 DLL 是硬依赖：不在 PATH 里，可执行文件根本起不来。
    foreach ($dir in @('D:\deps\ffmpeg\bin', 'D:\deps\curl\bin')) {
        if (Test-Path -LiteralPath $dir) {
            $env:PATH = "$dir;$env:PATH"
        } else {
            Write-Warn "缺少依赖目录 $dir —— 可执行文件可能因找不到 ffmpeg/libcurl 的 DLL 而无法启动。"
        }
    }
    if (-not (Test-Path -LiteralPath $script:TmpDir)) {
        [void](New-Item -ItemType Directory -Path $script:TmpDir -Force)
    }
}

function Assert-AppExecutables {
    foreach ($exe in @($script:ExeCli, $script:ExeServe)) {
        if (-not (Test-Path -LiteralPath $exe)) {
            throw "可执行文件不存在：$exe`n先构建（.\.workbuddy\tmp\msvcbuild.py targets 89 ninfer ninfer-serve）或用 -AppDir 指定其它目录。"
        }
    }
}

function Test-CrtDeployed {
    param([string]$Dir)
    foreach ($dll in @('msvcp140.dll', 'vcruntime140.dll', 'msvcp140_atomic_wait.dll')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Dir $dll))) { return $false }
    }
    return $true
}

function Invoke-DeployCrt {
    [CmdletBinding()]
    param([string]$Dir)

    $root = Split-Path -Parent $Dir
    $crtNames = @(
        'concrt140.dll', 'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
        'msvcp140_atomic_wait.dll', 'msvcp140_codecvt_ids.dll',
        'vcruntime140.dll', 'vcruntime140_1.dll', 'vcruntime140_threads.dll'
    )

    $vcRoots = @(
        'D:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC',
        'C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC',
        'D:\Program Files\Microsoft Visual Studio\2022\Professional\VC',
        'C:\Program Files\Microsoft Visual Studio\2022\Professional\VC',
        'D:\Program Files\Microsoft Visual Studio\2022\Community\VC',
        'C:\Program Files\Microsoft Visual Studio\2022\Community\VC'
    )

    $source = $null
    foreach ($vc in $vcRoots) {
        $redistRoot = Join-Path $vc 'Redist\MSVC'
        if (-not (Test-Path -LiteralPath $redistRoot)) { continue }
        $version = Get-ChildItem -LiteralPath $redistRoot -Directory |
            Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'x64\Microsoft.VC143.CRT') } |
            Sort-Object { try { [version]$_.Name } catch { [version]'0.0' } } |
            Select-Object -Last 1
        if ($null -ne $version) {
            $source = Join-Path $version.FullName 'x64\Microsoft.VC143.CRT'
            break
        }
    }
    if ($null -eq $source) {
        throw '找不到任何 x64 VC143 redist（Microsoft.VC143.CRT）。请确认 Visual Studio 2022 已安装 C++ 工具集。'
    }

    foreach ($target in @($Dir, (Join-Path $root 'tests'))) {
        if (-not (Test-Path -LiteralPath $target)) { continue }
        foreach ($name in $crtNames) {
            Copy-Item -LiteralPath (Join-Path $source $name) -Destination $target -Force
        }
        Write-Ok "CRT -> $target"
    }
    Write-Note "来源：$source"
}

function Initialize-CrtIfNeeded {
    if ($FixCrt) {
        Write-Say '重新部署 VC143 CRT ...'
        Invoke-DeployCrt -Dir $AppDir
        return
    }
    if (-not (Test-CrtDeployed -Dir $AppDir)) {
        Write-Warn "可执行文件旁边没有 VC143 CRT。System32 里那套是 2016 年的旧版，第一次加锁就会 0xC0000005。"
        Write-Say '正在自动部署 ...'
        Invoke-DeployCrt -Dir $AppDir
    }
}

function Resolve-ModelPath {
    if ($Model) {
        if (-not (Test-Path -LiteralPath $Model)) { throw "指定的模型不存在：$Model" }
        return (Resolve-Path -LiteralPath $Model).Path
    }
    if (-not (Test-Path -LiteralPath $ModelDir)) {
        throw "模型目录不存在：$ModelDir（用 -ModelDir 或 -Model 指定）"
    }
    $preferred = @(
        'qwen38-27b-q4v3mtp.ninfer',
        'qwen38-27b-q4v2.ninfer',
        'qwen38-27b-final.ninfer',
        'qwen35-9b-final.ninfer'
    )
    foreach ($name in $preferred) {
        $candidate = Join-Path $ModelDir $name
        if (Test-Path -LiteralPath $candidate) { return (Resolve-Path -LiteralPath $candidate).Path }
    }
    $any = Get-ChildItem -LiteralPath $ModelDir -Filter '*.ninfer' -ErrorAction SilentlyContinue |
        Sort-Object Length -Descending | Select-Object -First 1
    if ($null -ne $any) { return $any.FullName }
    throw "$ModelDir 下找不到任何 .ninfer 产物。用 -Model <路径> 显式指定。"
}

function Get-EffectiveModel([string]$Path) {
    # 9B 装得下 8GB 显存，不需要 offload；27B 必须 offload。
    $name  = [System.IO.Path]::GetFileName($Path)
    $is9B  = $name -match '9b'
    $ratio = $OffloadRatio
    if ($is9B -and (-not $script:OffloadRatioSpecified)) { $ratio = 0.0 }
    return [pscustomobject]@{
        Path    = $Path
        Name    = $name
        SizeGiB = [math]::Round((Get-Item -LiteralPath $Path).Length / 1GB, 2)
        Is9B    = $is9B
        Ratio   = $ratio
        HasMtp  = ($name -match 'mtp')
    }
}

# ---------------------------------------------------------------------------
# 2. 参数组装
# ---------------------------------------------------------------------------
function Write-Utf8File {
    param([string]$Path, [string]$Text)
    # 关键：不带 BOM 的 UTF-8。CLI 的 JSON 解析器见到 BOM 会直接失败。
    [System.IO.File]::WriteAllText($Path, $Text, (New-Object System.Text.UTF8Encoding($false)))
}

function New-MessageFile {
    param([string]$Text, [string]$Path)
    $payload = [pscustomobject]@{
        messages = @([pscustomobject]@{ role = 'user'; content = $Text })
    }
    Write-Utf8File -Path $Path -Text ($payload | ConvertTo-Json -Depth 8 -Compress)
    return $Path
}

function Resolve-PromptText {
    if ($PromptFile) {
        if (-not (Test-Path -LiteralPath $PromptFile)) { throw "提示词文件不存在：$PromptFile" }
        return [System.IO.File]::ReadAllText($PromptFile, [System.Text.Encoding]::UTF8).Trim()
    }
    if ($Prompt) { return $Prompt }
    return $null
}

function Get-NormalizedKv {
    if ($KvCapacity -lt $MaxContext) {
        Write-Warn "KvCapacity($KvCapacity) 小于 MaxContext($MaxContext)，已自动提升到 $MaxContext。"
        return $MaxContext
    }
    return $KvCapacity
}

function New-SamplingArgs {
    $result = New-Object System.Collections.Generic.List[string]
    if ($Greedy) {
        $result.Add('--greedy')
    } elseif ($script:TemperatureSpecified -and $Temperature -ge 0) {
        $result.Add('--temperature'); $result.Add((Get-Invariant $Temperature))
    }
    return $result.ToArray()
}

function New-CommonArgs {
    param([object]$Effective, [switch]$Quiet)

    $kv   = Get-NormalizedKv
    $list = New-Object System.Collections.Generic.List[string]
    $list.Add('--max-context'); $list.Add("$MaxContext")
    $list.Add('--kv-capacity'); $list.Add("$kv")
    $list.Add('--device');      $list.Add("$Device")
    if ($KvDtype) { $list.Add('--kv-dtype'); $list.Add($KvDtype) }
    if ($Effective.Ratio -gt 0) {
        $list.Add('--offload-ratio'); $list.Add((Get-Invariant $Effective.Ratio))
    }
    if (-not $NoSpec) {
        if ($Effective.HasMtp) {
            $list.Add('--spec'); $list.Add('mtp')
            $list.Add('--draft-tokens'); $list.Add("$DraftTokens")
        } else {
            Write-Warn "模型产物 $( $Effective.Name ) 不含 MTP 组件，已跳过投机解码。"
        }
    }
    if (-not $Thinking) { $list.Add('--no-thinking') }
    if ($Quiet) {
        # 后台服务跑在暗窗里，日志既看不到又会打断 repl 的流式显示，降到 warning 只留故障。
        $list.Add('--log-level'); $list.Add('warning')
    } elseif ($LogLevel) {
        $list.Add('--log-level'); $list.Add($LogLevel)
    }
    foreach ($extra in $ExtraArgs) { $list.Add($extra) }
    return $list.ToArray()
}

function Get-MemoryGiB {
    try {
        $raw = & nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $raw) { return $raw }
    } catch { }
    return $null
}

function Invoke-Native {
    <#
        在 $ErrorActionPreference = 'Stop' 之下，把原生命令的 stderr 并入管道会被 PowerShell 当成
        终止性错误（NativeCommandError）而中断整个脚本 —— 所以这里局部放宽。
        另外，PowerShell 会把原生 stderr 包装成 ErrorRecord，直接 Tee 进日志会得到一坨
        CategoryInfo / NativeCommandError 的渲染文本；用 ForEach-Object 把 ErrorRecord 还原成
        原始文本行即可。ForEach-Object 是流式 cmdlet，实时输出不受影响。
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [string]$LogPath = '',
        [switch]$CaptureAll
    )

    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'

    $flatten = {
        if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.TargetObject } else { $_ }
    }

    try {
        if ($CaptureAll) {
            $text = & $Exe @Arguments 2>&1 | ForEach-Object $flatten | Out-String
            return [pscustomobject]@{ ExitCode = $LASTEXITCODE; StdOut = $text; StdErr = '' }
        }

        if ($LogPath) {
            # Tee-Object 在 PowerShell 5.1 没有 -Encoding，落盘为 UTF-16LE（记事本/VS Code 均可读）。
            & $Exe @Arguments 2>&1 | ForEach-Object $flatten | Tee-Object -FilePath $LogPath
            return [pscustomobject]@{ ExitCode = $LASTEXITCODE; StdOut = ''; StdErr = '' }
        }

        & $Exe @Arguments
        return [pscustomobject]@{ ExitCode = $LASTEXITCODE; StdOut = ''; StdErr = '' }
    } finally {
        $ErrorActionPreference = $saved
    }
}

function Format-ExitCode {
    param([int]$Code)
    if ($Code -eq 0) { return '0' }
    $unsigned = [uint32]$Code
    return ("{0} (0x{1:X8})" -f $Code, $unsigned)
}

# ---------------------------------------------------------------------------
# 3. 模式实现
# ---------------------------------------------------------------------------
function Invoke-Doctor {
    Write-Host ''
    Write-Host 'NInfer 本地启动器 · 环境体检' -ForegroundColor White
    Write-Host ('-' * 62)

    # 可执行文件
    foreach ($exe in @($script:ExeCli, $script:ExeServe)) {
        if (Test-Path -LiteralPath $exe) {
            $item = Get-Item -LiteralPath $exe
            Write-Ok ("{0}  ({1:yyyy-MM-dd HH:mm})" -f $item.Name, $item.LastWriteTime)
        } else {
            Write-Bad "缺少 $exe"
        }
    }

    # CRT
    if (Test-CrtDeployed -Dir $AppDir) {
        $crt = Get-Item -LiteralPath (Join-Path $AppDir 'msvcp140.dll')
        Write-Ok ("VC143 CRT 已旁加载（msvcp140.dll {0:yyyy-MM-dd}）" -f $crt.LastWriteTime)
    } else {
        Write-Bad 'VC143 CRT 未旁加载 —— 运行时会 0xC0000005，请加 -FixCrt'
    }

    # 依赖 DLL
    foreach ($dir in @('D:\deps\ffmpeg\bin', 'D:\deps\curl\bin')) {
        if (Test-Path -LiteralPath $dir) { Write-Ok "依赖 DLL 目录存在：$dir" }
        else { Write-Bad "缺少依赖 DLL 目录：$dir" }
    }

    # 显存
    $gpu = Get-MemoryGiB
    if ($gpu) { Write-Ok "GPU：$gpu" } else { Write-Warn '无法调用 nvidia-smi，跳过显存检查。' }

    # 模型产物
    Write-Host ''
    Write-Host '可用模型产物：' -ForegroundColor White
    if (Test-Path -LiteralPath $ModelDir) {
        $models = Get-ChildItem -LiteralPath $ModelDir -Filter '*.ninfer' -ErrorAction SilentlyContinue |
            Sort-Object Length -Descending
        if ($models) {
            foreach ($m in $models) {
                $tag = ''
                if ($m.Name -eq 'qwen38-27b-q4v3mtp.ninfer') { $tag = '<- 推荐' }
                Write-Host ("  {0,-32} {1,7:N2} GiB  {2:yyyy-MM-dd HH:mm}  {3}" -f $m.Name, ($m.Length / 1GB), $m.LastWriteTime, $tag)
            }
        } else {
            Write-Bad "  $ModelDir 下没有任何 .ninfer 产物"
        }
    } else {
        Write-Bad "  模型目录不存在：$ModelDir"
    }

    # 推荐命令
    Write-Host ''
    Write-Host '推荐启动命令：' -ForegroundColor White
    Write-Host '  .\run-27b.bat chat -Prompt "用一句话介绍你自己。"' -ForegroundColor DarkGray
    Write-Host '  .\run-27b.bat web' -ForegroundColor DarkGray
    Write-Host ''
}

function Invoke-Chat {
    $effective = Get-EffectiveModel (Resolve-ModelPath)
    Write-Say "模型 $($effective.Name)（$($effective.SizeGiB) GiB）"

    $text = Resolve-PromptText
    if (-not $text) {
        throw '请用 -Prompt "..." 或 -PromptFile <UTF-8 文件> 提供提问内容。'
    }

    $msgPath = Join-Path $script:TmpDir 'chat-messages.json'
    [void](New-MessageFile -Text $text -Path $msgPath)

    $cliArgs = New-Object System.Collections.Generic.List[string]
    $cliArgs.Add($effective.Path)
    $cliArgs.Add('--messages'); $cliArgs.Add($msgPath)
    $cliArgs.Add('--max-new');  $cliArgs.Add("$MaxNew")
    foreach ($a in (New-SamplingArgs)) { $cliArgs.Add($a) }
    foreach ($a in (New-CommonArgs -Effective $effective)) { $cliArgs.Add($a) }

    Write-Note ("offload={0}  spec={1}  max-new={2}" -f $effective.Ratio, $(if ($NoSpec) { 'off' } else { "mtp k=$DraftTokens" }), $MaxNew)
    Write-Host ''

    $result = Invoke-Native -Exe $script:ExeCli -Arguments $cliArgs.ToArray() -LogPath $Log
    if ($Log) { Write-Note "输出已保存：$Log" }

    if ($result.ExitCode -ne 0) {
        if ([uint32]$result.ExitCode -eq 0xC0000409) {
            Write-Bad '进程以 0xC0000409 退出：算子路由被拒（多为模型量化格式与形状表不匹配）。'
        } else {
            Write-Bad "进程退出码 $(Format-ExitCode -Code $result.ExitCode)"
        }
    }
    return $result.ExitCode
}

function Invoke-Bench {
    $effective = Get-EffectiveModel (Resolve-ModelPath)

    $benchText = '请连续写十二句关于四季变化的话，一句一行，不要停，写满为止。'
    $msgPath   = Join-Path $script:TmpDir 'bench-messages.json'
    [void](New-MessageFile -Text $benchText -Path $msgPath)

    $runLog = Join-Path $script:TmpDir 'bench.log'

    $cliArgs = New-Object System.Collections.Generic.List[string]
    $cliArgs.Add($effective.Path)
    $cliArgs.Add('--messages'); $cliArgs.Add($msgPath)
    $cliArgs.Add('--max-new');  $cliArgs.Add("$MaxNew")
    $cliArgs.Add('--greedy')
    foreach ($a in (New-CommonArgs -Effective $effective)) { $cliArgs.Add($a) }

    Write-Say "基准运行：$($effective.Name)  max-new=$MaxNew  offload=$($effective.Ratio)"
    Write-Note '运行中（27B 约 1~3 分钟），完成后打印统计 ...'

    $capturedResult = Invoke-Native -Exe $script:ExeCli -Arguments $cliArgs.ToArray() -CaptureAll
    $captured = $capturedResult.StdOut + "`n" + $capturedResult.StdErr
    $captured | Set-Content -LiteralPath $runLog -Encoding UTF8

    if ($capturedResult.ExitCode -ne 0) {
        Write-Bad "进程退出码 $(Format-ExitCode -Code $capturedResult.ExitCode)，统计可能不完整。"
    }

    $metrics = [ordered]@{
        'prompt tokens'  = '-'
        'generated tokens' = '-'
        'model elapsed'  = '-'
        'prefill speed'  = '-'
        'decode speed'   = '-'
        'mtp acceptance rate' = '-'
        # 引擎的 summary 里这一行叫 "mtp acceptance length"，值形如 "2.04 tok/round"。
        # 早期版本这里写成 'mtp tok/round'，永远匹配不上、恒显示 '-'。
        'mtp acceptance length' = '-'
        'gpu weights used' = '-'
    }
    foreach ($key in @($metrics.Keys)) {
        $m = [regex]::Match($captured, ('^summary\s+' + [regex]::Escape($key) + '\s+(.+?)\s*$'), 'Multiline')
        if ($m.Success) { $metrics[$key] = $m.Groups[1].Value.Trim() }
    }

    Write-Host ''
    Write-Host '基准结果' -ForegroundColor White
    Write-Host ('-' * 44)
    foreach ($key in $metrics.Keys) {
        Write-Host ("  {0,-22} {1}" -f $key, $metrics[$key])
    }
    Write-Host ''
    Write-Note "完整日志：$runLog"
}

function New-ContextCacheArgs {
    <#
        serve 默认要 pin 8 GiB 主机 KV 用于跨请求的上下文缓存。offload 已经把约 12 GiB 的权重压在
        主机侧（同样是 pin 内存），两者叠加必然 cudaMallocHost OOM —— 实测报
        「pinning host KV | cudaMallocHost failed: cudaErrorMemoryAllocation」。
        因此 offload 打开时默认把 host KV 关掉（0），需要时用 -HostKvMiB 覆盖。
    #>
    param([object]$Effective)

    $list = New-Object System.Collections.Generic.List[string]
    $mib  = $HostKvMiB
    if ($mib -lt 0) {
        if ($Effective.Ratio -gt 0) {
            $mib = 0
            Write-Note 'offload 已占用主机内存，host KV 缓存自动设为 0（-HostKvMiB 可覆盖）。'
        } else {
            return $list.ToArray()
        }
    }
    $list.Add('--host-kv-mib'); $list.Add("$mib")
    return $list.ToArray()
}

function Quote-CommandLine {
    param([string[]]$Values)
    $out = foreach ($v in $Values) {
        if ($v -match '\s') { '"' + $v + '"' } else { $v }
    }
    return ($out -join ' ')
}

function Start-ServeBackground {
    param([object]$Effective, [switch]$Cors)

    $serveArgs = New-Object System.Collections.Generic.List[string]
    $serveArgs.Add($Effective.Path)
    $serveArgs.Add('--host');            $serveArgs.Add($BindHost)
    $serveArgs.Add('--port');            $serveArgs.Add("$Port")
    $serveArgs.Add('--max-concurrency'); $serveArgs.Add("$MaxConcurrency")
    foreach ($a in (New-CommonArgs -Effective $Effective -Quiet)) { $serveArgs.Add($a) }
    foreach ($a in (New-ContextCacheArgs -Effective $Effective)) { $serveArgs.Add($a) }
    $serveArgs.Add('--log-stats-interval-ms'); $serveArgs.Add('0')
    if ($Cors)   { $serveArgs.Add('--cors') }
    if ($ApiKey) { $serveArgs.Add('--api-key'); $serveArgs.Add($ApiKey) }

    Write-Say "启动本地服务 http://$BindHost`:$Port（KV 容量 $(Get-NormalizedKv) token，并发 $MaxConcurrency）"

    # 这里不能用 Start-Process：PowerShell 5.1 在启动子进程时会重建一份环境字典，而环境块里只要
    # 存在仅大小写不同的重复变量（Path/PATH、http_proxy/HTTP_PROXY 之类，代理环境里很常见），
    # 它就抛「已添加项。字典中的关键字…」而拒绝启动。System.Diagnostics.Process 让子进程直接继承
    # 父进程的环境块、不做重建，因此不受影响。
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName         = $script:ExeServe
    $psi.Arguments        = (Quote-CommandLine -Values $serveArgs.ToArray())
    $psi.UseShellExecute  = $false
    $psi.CreateNoWindow   = $true
    $psi.WorkingDirectory = $script:RepoRoot

    $process = [System.Diagnostics.Process]::Start($psi)
    return [pscustomobject]@{ Process = $process }
}

function Wait-ServeReady {
    param([object]$Handle, [int]$TimeoutSec = 300)

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $base = "http://$BindHost`:$Port"
    Write-Note '模型加载中（27B 约 10~60 秒），等待服务就绪 ' -NoNewline

    while ((Get-Date) -lt $deadline) {
        if ($Handle.Process.HasExited) {
            Write-Host ''
            Write-Bad "服务进程已退出（exit $(Format-ExitCode -Code $Handle.Process.ExitCode)）。"
            Write-Note "换 run-27b.ps1 serve 前台跑一次，就能看到完整启动日志。"
            throw '服务启动失败。'
        }
        try {
            $resp = Invoke-WebRequest -Uri "$base/health" -UseBasicParsing -TimeoutSec 3
            if ($resp.StatusCode -eq 200) {
                Write-Host ''
                Write-Ok "服务就绪：$base"
                return $true
            }
        } catch { }
        Write-Host '.' -NoNewline
        Start-Sleep -Milliseconds 800
    }
    Write-Host ''
    throw "等待服务就绪超时（$TimeoutSec 秒）。"
}

function Stop-ServeHandle {
    param([object]$Handle)
    if ($null -eq $Handle) { return }
    if ($Handle.Process -and (-not $Handle.Process.HasExited)) {
        Write-Note '正在停止本地服务 ...'
        Stop-Process -Id $Handle.Process.Id -Force -ErrorAction SilentlyContinue
    }
}

function Get-PublicModelId {
    param([string]$BaseUrl, [string]$Key)
    try {
        $headers = @{}
        if ($Key) { $headers['Authorization'] = "Bearer $Key" }
        $resp = Invoke-RestMethod -Uri "$BaseUrl/v1/models" -Headers $headers -TimeoutSec 10
        $first = @($resp.data)[0]
        if ($first -and $first.id) { return $first.id }
    } catch { }
    return 'local'
}

function Invoke-ChatStream {
    param([string]$BaseUrl, [string]$Json, [string]$Key, [switch]$ShowReasoning)

    try { Add-Type -AssemblyName System.Net.Http -ErrorAction SilentlyContinue } catch { }

    $client = New-Object -TypeName System.Net.Http.HttpClient
    $client.Timeout = [TimeSpan]::FromMinutes(60)
    try {
        $request = New-Object -TypeName System.Net.Http.HttpRequestMessage -ArgumentList @(
            [System.Net.Http.HttpMethod]::Post, "$BaseUrl/v1/chat/completions")
        $request.Content = New-Object -TypeName System.Net.Http.StringContent -ArgumentList @(
            $Json, [System.Text.Encoding]::UTF8, 'application/json')
        if ($Key) { [void]$request.Headers.Add('Authorization', "Bearer $Key") }

        $response = $client.SendAsync($request, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
        if (-not $response.IsSuccessStatusCode) {
            $detail = $response.Content.ReadAsStringAsync().Result
            throw ("HTTP {0} {1}: {2}" -f [int]$response.StatusCode, $response.ReasonPhrase, $detail)
        }

        $stream = $response.Content.ReadAsStreamAsync().Result
        $reader = New-Object System.IO.StreamReader -ArgumentList @($stream, [System.Text.Encoding]::UTF8)
        $builder = New-Object System.Text.StringBuilder
        try {
            while ($true) {
                $line = $reader.ReadLine()
                if ($null -eq $line) { break }
                if (-not $line.StartsWith('data:')) { continue }
                $payload = $line.Substring(5).Trim()
                if ($payload -eq '[DONE]') { break }
                if ($payload.Length -eq 0) { continue }

                $event = $null
                try { $event = ConvertFrom-Json -InputObject $payload } catch { continue }
                $choices = @($event.choices)
                if ($choices.Count -eq 0) { continue }
                $delta = $choices[0].delta
                if ($null -eq $delta) { continue }

                if ($delta.reasoning_content -and $ShowReasoning) {
                    Write-Host $delta.reasoning_content -NoNewline -ForegroundColor DarkGray
                }
                if ($delta.content) {
                    Write-Host $delta.content -NoNewline
                    [void]$builder.Append($delta.content)
                }
            }
        } finally {
            $reader.Dispose(); $stream.Dispose(); $response.Dispose()
        }
        return $builder.ToString()
    } finally {
        $client.Dispose()
    }
}

function Start-ReplSession {
    param([string]$BaseUrl, [string]$Key, [string]$ModelId, [switch]$ShowReasoning)

    $history = New-Object System.Collections.Generic.List[object]

    Write-Host ''
    Write-Host '多轮对话就绪（权重只加载了一次）。' -ForegroundColor White
    Write-Host '  :exit / :q         退出' -ForegroundColor DarkGray
    Write-Host '  :reset             清空上下文' -ForegroundColor DarkGray
    Write-Host '  :file <路径>       从 UTF-8 文本文件读入提问（中文输入异常时的兜底）' -ForegroundColor DarkGray
    Write-Host '  :help              帮助' -ForegroundColor DarkGray

    while ($true) {
        Write-Host ''
        Write-Host '你 > ' -NoNewline -ForegroundColor Green
        $line = Read-Host
        if ($null -eq $line) { break }
        $line = $line.Trim()
        if ($line.Length -eq 0) { continue }

        if ($line -eq ':exit' -or $line -eq ':q') { break }
        if ($line -eq ':reset') {
            $history.Clear()
            Write-Note '上下文已清空。'
            continue
        }
        if ($line -eq ':help') {
            Write-Host '  输入问题回车即可；模型回答逐字流式显示。' -ForegroundColor DarkGray
            Write-Host '  :file 用法示例：:file D:\deps\q.txt' -ForegroundColor DarkGray
            continue
        }
        if ($line.StartsWith(':file ')) {
            $path = $line.Substring(6).Trim().Trim('"')
            if (-not (Test-Path -LiteralPath $path)) {
                Write-Warn "文件不存在：$path"
                continue
            }
            $line = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8).Trim()
            Write-Note ("已从文件读入 {0} 字符。" -f $line.Length)
        }

        $history.Add([pscustomobject]@{ role = 'user'; content = $line })

        $body = [ordered]@{
            model      = $ModelId
            messages   = $history.ToArray()
            stream     = $true
            max_tokens = $MaxNew
        }
        if ($Greedy) {
            $body['temperature'] = 0
        } elseif ($script:TemperatureSpecified -and $Temperature -ge 0) {
            $body['temperature'] = $Temperature
        }
        $json = $body | ConvertTo-Json -Depth 10 -Compress

        Write-Host 'AI > ' -NoNewline -ForegroundColor Cyan
        try {
            $answer = Invoke-ChatStream -BaseUrl $BaseUrl -Json $json -Key $Key -ShowReasoning:$ShowReasoning
            Write-Host ''
            $history.Add([pscustomobject]@{ role = 'assistant'; content = $answer })
        } catch {
            Write-Host ''
            Write-Bad "请求失败：$($_.Exception.Message)"
            [void]$history.RemoveAt($history.Count - 1)
        }
    }
}

function Invoke-Repl {
    $effective = Get-EffectiveModel (Resolve-ModelPath)
    $handle = $null
    try {
        $handle = Start-ServeBackground -Effective $effective
        [void](Wait-ServeReady -Handle $handle)
        $base = "http://$BindHost`:$Port"
        $modelId = Get-PublicModelId -BaseUrl $base -Key $ApiKey
        Start-ReplSession -BaseUrl $base -Key $ApiKey -ModelId $modelId -ShowReasoning:$Thinking
    } finally {
        Stop-ServeHandle -Handle $handle
    }
}

function Invoke-ServeForeground {
    $effective = Get-EffectiveModel (Resolve-ModelPath)
    $kv = Get-NormalizedKv

    $serveArgs = New-Object System.Collections.Generic.List[string]
    $serveArgs.Add($effective.Path)
    $serveArgs.Add('--host');            $serveArgs.Add($BindHost)
    $serveArgs.Add('--port');            $serveArgs.Add("$Port")
    $serveArgs.Add('--max-concurrency'); $serveArgs.Add("$MaxConcurrency")
    foreach ($a in (New-CommonArgs -Effective $effective)) { $serveArgs.Add($a) }
    foreach ($a in (New-ContextCacheArgs -Effective $effective)) { $serveArgs.Add($a) }
    if ($ApiKey) { $serveArgs.Add('--api-key'); $serveArgs.Add($ApiKey) }

    Write-Say "OpenAI 兼容端点在 http://$BindHost`:$Port/v1 ，Anthropic 端点在 /v1/messages"
    Write-Note '按 Ctrl+C 停止。'
    Write-Host ''
    $result = Invoke-Native -Exe $script:ExeServe -Arguments $serveArgs.ToArray()
    return $result.ExitCode
}

function Invoke-Web {
    $effective = Get-EffectiveModel (Resolve-ModelPath)
    $page = Join-Path $script:RepoRoot 'tools\local_chat\index.html'
    if (-not (Test-Path -LiteralPath $page)) { throw "网页聊天页不存在：$page" }

    $handle = $null
    try {
        $handle = Start-ServeBackground -Effective $effective -Cors
        [void](Wait-ServeReady -Handle $handle)

        $url = 'file:///' + ($page -replace '\\', '/')
        $url += "?port=$Port&host=$BindHost"
        if ($ApiKey) { $url += "&key=$([uri]::EscapeDataString($ApiKey))" }

        if (-not $NoOpen) {
            Write-Say '正在打开浏览器 ...'
            Start-Process $url
        } else {
            Write-Note "浏览器未自动打开，请手动访问：$url"
        }

        if ($Detach) {
            Write-Ok "服务已在后台运行（PID $($handle.Process.Id)）。停止：.\run-27b.bat stop"
            return
        }

        Write-Host ''
        Write-Note '服务运行中，按 Ctrl+C 或关闭窗口即停止。'
        while (-not $handle.Process.HasExited) { Start-Sleep -Milliseconds 400 }
    } finally {
        if (-not $Detach) { Stop-ServeHandle -Handle $handle }
    }
}

function Invoke-Stop {
    $procs = Get-Process -Name 'ninfer-serve' -ErrorAction SilentlyContinue
    if (-not $procs) {
        Write-Note '没有正在运行的 ninfer-serve 进程。'
        return
    }
    foreach ($p in $procs) {
        Write-Say "停止 ninfer-serve (PID $($p.Id)) ..."
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Ok '已停止。'
}

# ---------------------------------------------------------------------------
# 4. 分发
# ---------------------------------------------------------------------------
Initialize-Environment

switch ($Mode) {
    'doctor' { Invoke-Doctor; exit 0 }
    'stop'   { Invoke-Stop;   exit 0 }
    default  {
        Assert-AppExecutables
        Initialize-CrtIfNeeded
    }
}

switch ($Mode) {
    'chat'  { exit (Invoke-Chat) }
    'bench' { Invoke-Bench; exit 0 }
    'serve' { exit (Invoke-ServeForeground) }
    'repl'  { Invoke-Repl;  exit 0 }
    'web'   { Invoke-Web;   exit 0 }
}
