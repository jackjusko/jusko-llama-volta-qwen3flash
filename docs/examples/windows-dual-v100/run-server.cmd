@echo off
setlocal EnableExtensions
cd /d H:\Ling-3.0-flash\qwen3.8-flash-next
call "H:\Ling-3.0-flash\qwen3.8-flash-next\run-env.cmd"

set "BIN=%LLAMA_BIN_DIR%\llama-server.exe"
if not exist "%BIN%" (
  echo llama-server.exe is not built yet.
  exit /b 1
)
if not exist "%LLAMA_MODEL%" (
  echo Model shard 1 is missing: %LLAMA_MODEL%
  echo Run H:\Ling-3.0-flash\qwen3.8-flash-next\download.cmd first.
  exit /b 1
)

rem 110 GB mmap (54 GB of it page-cache-only table) + a leftover server
rem would thrash 128 GB RAM.
taskkill /F /IM llama-server.exe >nul 2>&1

rem This AtomicChat quant has no MTP tensors. llama.cpp exits if draft-mtp
rem is requested. Default is no spec (ngram drafts slow this hybrid). SPEC=1 is MTP.
if "%SPEC%"=="1" (
  set "SPEC_ARGS=--spec-type draft-mtp --spec-draft-n-max 3"
) else if "%SPEC%"=="0" (
  set "SPEC_ARGS="
) else (
  set "SPEC_ARGS=%LLAMA_SPEC_ARGS%"
)

rem Qwen thinking: chat-template effort + deepseek extraction.
rem Set REASONING=0 to disable. Default effort is medium.
if not defined REASONING_EFFORT set "REASONING_EFFORT=medium"
if "%REASONING%"=="0" (
  set "REASON_ARGS="
) else (
  set "REASON_ARGS=--reasoning on --reasoning-format deepseek --reasoning-effort %REASONING_EFFORT%"
)

"%BIN%" ^
  -m "%LLAMA_MODEL%" ^
  --alias qwen3.8-flash-next ^
  --jinja ^
  %REASON_ARGS% ^
  --temp 1.0 ^
  --top-p 0.95 ^
  --top-k 20 ^
  --min-p 0 ^
  %SPEC_ARGS% ^
  -c %LLAMA_CTX% ^
  %LLAMA_GPU_ARGS% ^
  %LLAMA_CPU_ARGS% ^
  %LLAMA_SRV_ARGS% ^
  --metrics ^
  --host 127.0.0.1 ^
  --port %LLAMA_PORT% %*
endlocal
