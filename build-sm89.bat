@echo off
REM ---------------------------------------------------------------------------
REM Build NInfer for a non-Blackwell target (sm_89 / Ada, e.g. RTX 4060 8GB).
REM
REM The NVFP4 w4a4 kernels and their TMA / fp4-value KV-cache companions need
REM ptxas features that first appear in sm_100a, so they are excluded here and
REM their host entry points fail loudly instead of silently degrading.
REM
REM Usage:  build-sm89.bat [configure|build|clean]
REM Override VS/CUDA locations with the VSDEVCMD and CUDA_PATH env vars.
REM ---------------------------------------------------------------------------
setlocal
set "REPO=%~dp0"
set "BUILD=%REPO%build-sm89"

if not defined VSDEVCMD set "VSDEVCMD=D:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"

if /i "%~1"=="clean" (
  if exist "%BUILD%" rmdir /s /q "%BUILD%"
  echo Removed %BUILD%
  exit /b 0
)

if not exist "%VSDEVCMD%" (
  echo ERROR: vcvars64.bat not found at "%VSDEVCMD%"
  echo Set VSDEVCMD to your Visual Studio VC\Auxiliary\Build\vcvars64.bat
  exit /b 1
)

call "%VSDEVCMD%" >nul
if errorlevel 1 ( echo ERROR: vcvars64.bat failed & exit /b 1 )

set "PATH=%CUDA_PATH%\bin;%PATH%"

if /i "%~1"=="configure" goto configure
if /i "%~1"=="build" goto build
goto configure

:configure
cmake -S "%REPO%." -B "%BUILD%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH:\=/%/bin/nvcc.exe" ^
  -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
  -DNINFER_BUILD_APPS=ON ^
  -DBUILD_TESTING=OFF ^
  -DNINFER_BUILD_BENCHMARKS=OFF
if errorlevel 1 ( echo CONFIGURE FAILED & exit /b 1 )
if /i "%~1"=="configure" exit /b 0

:build
cmake --build "%BUILD%" --parallel
if errorlevel 1 ( echo BUILD FAILED & exit /b 1 )
echo BUILD OK: %BUILD%
exit /b 0
