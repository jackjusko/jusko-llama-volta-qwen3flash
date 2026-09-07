@echo off
rem Shared env for Qwen3.8-Flash-Next on this machine:
rem   GPU: V100 32GB + V100 32GB only (P40 hidden). NVLink P2P between the SXM2s.
rem   CPU: 1x E5-2697 v4, 18 cores / 36 threads, AVX2+FMA, no AVX-512
rem   RAM: 8x16GB DDR4-2400 ECC, 128 GB
rem   Binary: Ling V100 llama-server (sm70+sm61, FORCE_MMQ, qwen4exp, Volta GDN)
rem
rem Model: AtomicChat/Qwen3.8-Flash-Next-GGUF AD-5.00bpw-Q5_K_M-M64
rem   total on disk   110.5 GB (33 shards; shard 2 is ONLY the n-gram table)
rem   in-memory part  56.1 GB  (48 layers + MTP, GPU-resident)
rem   n-gram table    54.4 GB  (lazy mmap, host-gathered, never copied to VRAM)
rem
rem AtomicChat: --fit mis-sizes this architecture. Use -ngl 99 --fit off.
rem Weights (~52 GiB) fit on 2x V100 (64 GB). KV at 200k is ~6 GB.
rem PLE (~54 GiB) is host-only: --lazy-mode off (resident). Never mmap+mlock the whole GGUF.

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
set "PATH=%CUDA_PATH%\bin;%PATH%"
rem Sweep scripts may override these before calling run-server.cmd.
if not defined CUDA_VISIBLE_DEVICES set "CUDA_VISIBLE_DEVICES=0,1"
rem Direct GPU-GPU copies over NVLink (V100-SXM2). Must be set to any value.
if not defined GGML_CUDA_P2P set "GGML_CUDA_P2P=1"

rem Cap OpenMP at physical cores. ggml already uses -t 18; 36 OMP workers would
rem fight HT siblings and cut quad-channel bandwidth.
set "OMP_NUM_THREADS=18"
set "OMP_WAIT_POLICY=ACTIVE"
set "OMP_PROC_BIND=close"
set "OMP_PLACES=cores"

if not defined LLAMA_BIN_DIR set "LLAMA_BIN_DIR=H:\Ling-3.0-flash\llama.cpp\build\bin"
set "LLAMA_MODEL=H:\Ling-3.0-flash\qwen3.8-flash-next\models\Qwen3.8-Flash-Next-AD-5.00bpw-Q5_K_M-M64\Qwen3.8-Flash-Next-AD-5.00bpw-Q5_K_M-M64-00001-of-00033.gguf"

if not defined LLAMA_CTX set "LLAMA_CTX=163840"
if not defined LLAMA_PORT set "LLAMA_PORT=8080"

rem Physical cores only, strict pin, poll for low-latency dispatch.
rem PLE (~54 GiB) must stay resident: --lazy-mode auto reads it from H: on demand (~45 t/s PP).
rem --lazy-mode off loads the table into host RAM once. Do NOT use mmap+mlock: that mlocks
rem every shard (~110 GiB) and blows past 128 GB system RAM (weights already live in VRAM).
rem LLAMA_MLOCK_PLE=1 VirtualLocks only TENSOR_READ_LAZY tables (~51 GiB PLE), so Windows
rem cannot discard those mmap pages under memory pressure.
set "LLAMA_MLOCK_PLE=1"
set "LLAMA_CPU_ARGS=-t 18 -tb 18 --cpu-mask 0x555555555 --cpu-strict 1 --poll 1 --prio 2 --load-mode mmap --lazy-mode off"

rem Full offload of the 56 GB transformer. --fit on fails on qwen4exp.
rem moe-cache off: experts are GPU-resident, nothing to park in RAM.
rem Frozen: 160k ctx, layer split, F16 KV (no q8).
rem 160k + ub 1024/2048 OOMs compute buffers (~2.8/5.6 GiB). ub 768 + pipeline 1 fits.
rem Volta: prefill FA still MMA-F16; decode still TILE. NVLink idle (layer ping-pong).
rem Long-ctx: QSA slim (block top-k) is default on; gather stays off (lost TG-2048).
rem Unmeasured remaining levers (leave unset / 0):
rem   QWEN4EXP_QSA_GATHER_AUTO=1  — gather FA only when n_kv >= QWEN4EXP_QSA_GATHER_MIN_KV (8192)
rem   QWEN4EXP_QSA_POOLED_K=1     — running indexer-K; scores from V only if n_kv >= POOLED_K_MIN_KV (8192)
rem   QWEN4EXP_QSA_FUSED_LID=1    — 4-head lightning indexer; skipped below FUSED_LID_MIN_BLOCKS (2048)
rem   QWEN4EXP_HC_PRE=1 / QWEN4EXP_HC_COMB_BCAST=1 — fused HC mean / broadcast combine
rem   QWEN4EXP_KV_COMPACT=1       — pack KV cells after seq_rm so used_max_p1 == used
rem q8 KV not used (user request). P40 moe-cache / ncmoe / DECODE_MMA: previously killed.
if not defined QWEN4EXP_QSA_SLIM set "QWEN4EXP_QSA_SLIM=1"
set "LLAMA_GPU_ARGS=-ngl 99 -fa on -sm layer -np 1 -b 1536 -ub 768 --prefill-reuse 768 --pipeline-copies 1 --cache-type-k f16 --cache-type-v f16 --fit off --moe-cache off"
if not defined LLAMA_SPEC_ARGS set "LLAMA_SPEC_ARGS="
rem Do not keep a poisoned KV high-water from prior prompts. QSA n_kv follows used_max_p1.
if not defined LLAMA_SRV_ARGS set "LLAMA_SRV_ARGS=--cache-ram 0 --slot-prompt-similarity 0"
