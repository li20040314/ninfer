@echo off
REM ---------------------------------------------------------------------------
REM NInfer local launcher for Qwen3.8-27B on an 8GB Ada card (RTX 4060 Laptop).
REM
REM Thin wrapper around run-27b.ps1: it only bypasses the execution policy and
REM forwards every argument, so the launcher works from cmd.exe and by
REM double-click, not just from a PowerShell prompt.
REM
REM Usage:  run-27b.bat [doctor|chat|repl|serve|web|bench|stop] [options]
REM   e.g.  run-27b.bat doctor
REM         run-27b.bat chat -Prompt "hello"
REM         run-27b.bat web
REM
REM Full guide: doc\rtx4060-27b-local-launch-guide.md
REM ---------------------------------------------------------------------------
setlocal
set "SCRIPT=%~dp0run-27b.ps1"
if not exist "%SCRIPT%" (
  echo ERROR: run-27b.ps1 not found next to this wrapper.
  pause
  exit /b 1
)

if "%~1"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" doctor
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
)
set "CODE=%ERRORLEVEL%"

REM Launched by double-click (no arguments) the console window would vanish
REM before the report can be read, so hold it open in that case only.
if "%~1"=="" (
  echo.
  pause
)
exit /b %CODE%
