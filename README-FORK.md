# This fork: 2x V100 + Qwen3.8-Flash-Next layer

The base of this tree is [`mistrjirka/llama.cpp`](https://github.com/mistrjirka/llama.cpp)
(branch `v100-optimized`, base commit `ac8d815`) — a CUDA performance fork for NVIDIA
Volta (SM70) and Turing (SM75), whose kernels were never fully documented outside its
author's handoff notes ([`HANDOFF_2026-09-05_SM70_SM75.md`](HANDOFF_2026-09-05_SM70_SM75.md)).

On top of that base this tree carries a second, separate optimization layer built for
**Qwen3.8-Flash-Next** (arch `qwen4exp`) served on **2x Tesla V100-SXM2 32 GB over
NVLink**. That layer is the bulk of the local changes: ~1,870 inserted lines across 27
modified files plus 3 new files (`ggml-backend-moe-cache.h`, `ggml-cuda/moe-cache.cu/.cuh`).

This file answers two questions:

1. **What is Volta-optimized** — inherited from the base fork, vs plain llama.cpp.
2. **What is Qwen3.8-Flash-Next-optimized** — added in this tree, on top of both plain
   llama.cpp *and* the base fork's Volta stack.

Benchmark evidence lives in [`docs/fork-benchmarks.md`](docs/fork-benchmarks.md).
The exact production launch config is in
[`docs/examples/windows-dual-v100/`](docs/examples/windows-dual-v100/).

---

## Daily-driver hardware / build

| Part | Spec |
|---|---|
| GPU | 2x Tesla V100-SXM2 32 GB, NVLink P2P (`GGML_CUDA_P2P=1`); a P40 exists in the box but is hidden from the Qwen driver (`CUDA_VISIBLE_DEVICES=0,1`) |
| CPU | 1x Intel Xeon E5-2697 v4, 18 cores / 36 threads, AVX2+FMA, no AVX-512 |
| RAM | 128 GB DDR4-2400 ECC, quad channel (~76.8 GB/s) |
| CUDA | 12.2, MSVC; build reports `ARCHS = 610,700` (Pascal 6.1 + Volta 7.0 native) \| `FORCE_MMQ=1` \| `USE_GRAPHS=1` \| `REPACK=1` \| `OPENMP=1` |

The model: **`Qwen3.8-Flash-Next-AD-5.00bpw-Q5_K_M-M64`**
(AtomicChat GGUF, 33 shards, 110.5 GB on disk):

- 48 transformer layers: **12 QSA block-sparse attention** layers (every 4th,
  `compress_ratio=4`, `indexer_top_k=2048`) + **36 GDN gated-delta-net linear** layers.
- Sparse MoE, **10-of-512** routed experts, hyper-connections (HC), and a
  **per-layer token embedding table (PLE)**: ~51.8 GiB, CPU-only by design.
- In-GPU transformer part: ~52-56 GiB → fits 2x V100. KV at 200k is ~6 GB (F16).
- **Shard 2 is only the n-gram draft table** (~54.4 GiB): lazy-mmap'd, host-gathered,
  never copied to VRAM ("n-gram offloaded").
- This quant carries **no MTP tensors**; `--spec-type draft-mtp` exits. Speculation is
  off by default (n-gram drafts lost on unique text — see benchmarks).

---

## Part A — Volta (SM70) optimizations inherited from the base fork

All of this is already in the base tree (`git show ac8d815`); the upstream README only
partially documents it. Against plain llama.cpp:

### A1. FlashAttention for Volta's tensor cores (no `cp.async`)

- `VOLTA_MMA_AVAILABLE` is defined for SM70 (`ggml/src/ggml-cuda/common.cuh`), enabling
  the f16 MMA FlashAttention path (`fattn-mma-f16.cuh`) that plain llama.cpp gates off
  on Volta. Kernels use explicit `mma.sync.aligned.m16n8k16` with register-permutation
  control (`mma.cuh`, `GGML_CUDA_MMA_NO_VOLTA_PERM` toggle).
- **256x256 FA tiling config with reduced V scratch** plus an **adaptive 2-CTA
  specialization** for the exact Qwen attention geometry (documented in the base
  `README.md` "Implementation" section; measurements there: +27.8 % PP on
  100k-cached Qwen3.8-27B, single V100).
- **q8_0-KV tensor-core attention sub-kernel**:
  `GGML_CUDA_VOLTA_Q8_FATTN_TC=1` selects `BEST_FATTN_KERNEL_VOLTA_Q8_W4`
  (`fattn.cu`). Measured ~2.674 ms -> ~1.419 ms at ~101k KV,
  ~6.879 ms -> ~3.363 ms at ~260k KV.
- **Exact long-K Qwen GQA8 geometry**: `GGML_CUDA_VOLTA_GQA8_NCOLS2=2`
  (D=256/GQA8 dispatch option in `fattn.cu`).
- Prefill uses cuBLAS `TENSOR_OP` + FA MMA-F16 when Q is wide; decode FA on this arch
  lands on **TILE** (`Q x GQA_eff <= 16`, `nbatch_fa=64`).

### A2. Volta GEMV / MMQ paths (memory-bound decode matmuls)

- `GGML_CUDA_VOLTA_FORCE_MMQ=moe` — opt-in routed-MoE MMQ override (leave unset for
  dense Qwen; Q5_K under `FORCE_MMQ` decodes via DP4A, not TC).
- Large-row Q6_K MMQ uses the **Pascal DP4A configuration** to cut register pressure.
- `GGML_CUDA_VOLTA_Q5_X4=1`, `GGML_CUDA_VOLTA_Q6_W4R4=1` — Q5 x4 and Q6
  warp4/row4 GEMV variants tuned for V100 HBM.

### A3. Hybrid-arch kernels

- **Volta scalar-gate GatedDeltaNet x4 prefill** path (GDN linear-attention layers;
  verified 40/40 on V100).
- **Qwen q8 attention and MTP generation paths**; **prompt checkpoint/replay** for
  recurrent state (`+11–14 %` PP on suffix appends at 100k), and
  `--spec-mtp-defer-prompt` (deferred MTP prompt catch-up, ~5–6 % TTFT).

### A4. Backend / multi-GPU plumbing (base)

- Mirrored-input copy path: large contiguous host inputs go through the backend async
  setter + one sync (copy fan-out reduction).
- `--prefill-reuse` (CUDA prefill GEMM tile weight reuse), `--pipeline-copies`,
  CUDA graphs (`USE_GRAPHS=1`), `GGML_CUDA_P2P` for NVLink (upstream feature, required
  here).

### A5. Turing (SM75) — inherited, not used by the V100 driver

Turing GDN x4, D=256 smaller prompt tiles, `GGML_CUDA_TURING_CUBLAS_MIN_BATCH`,
GQA6 `ncols2=2`, exact-geometry rescale skip. See `HANDOFF_2026-09-05_SM70_SM75.md`.

> **Rule from the base author:** SM70 and SM75 paths are separate and geometry-gated.
> `GGML_CUDA_VOLTA_*` controls must stay unset on SM75.

---

## Part B — what this tree adds for Qwen3.8-Flash-Next

Everything below is local work (uncommitted in this working tree; see "Commit
manifest"). All new Qwen knobs are env-gated and default **off** unless stated.

### B1. QSA block-sparse indexer, four ways (`src/models/qwen4exp.cpp`)

The base graph scored the indexer unfused (`mul_mat`/`get_rows`/`top_k`) over `n_kv`,
making decode O(n_kv) on the 12 QSA layers. This layer adds alternative graph shapes:

| Env | Mode | Mechanism | Status (2x V100) |
|---|---|---|---|
| `QWEN4EXP_QSA_SLIM=1` | **default ON** in the driver | block-level top-k: CAST winning block ids to F32 + `GET_ROWS` of only the winning blocks, expand + slice to `width` (2051); indexer scores `n_blocks` (~513), not `n_kv` | promoted: TG-20k 28.8 -> 29.9, TG-60k 22.1 -> 23.7, TG-150k 14.8 -> 16.1, think-512 34.5 -> 35.1 |
| `QWEN4EXP_QSA_GATHER=1` | off | dense TILE below `QWEN4EXP_QSA_GATHER_MIN_KV` (8192); block `GET_ROWS` of r-row KV chunks above (O(width) FA over ~2k cells). Layer-split only | killed: lost TG-2048 (32.2 vs 34.4) and cancelled slim's 20k gain (H04); `GATHER_AUTO` retested at 60k and killed (L01, 22.2) |
| `QWEN4EXP_QSA_POOLED_K=1` | off | running pooled indexer-K; score from V-cache only above `QWEN4EXP_QSA_POOLED_K_MIN_KV` (8192); decode tail-updates V. ~4x less indexer HBM at long ctx | off: wins 60k/150k (24.7 / 17.6) but -2k tax (33.4-33.7); length gate implemented, needs re-measure |
| `QWEN4EXP_QSA_FUSED_LID=1` | off | fused 4-head lightning-indexer vec kernel (Volta has no Turing WMMA -> vec-only, `N_HEAD_INNER=1`), gated by `QWEN4EXP_QSA_FUSED_LID_MIN_BLOCKS` (2048) | off: 33.5/29.0 vs 34.4/28.8 as always-on; kernel kept (`lightning-indexer.cu`) |
| `QWEN4EXP_QSA_LOG=1` | diagnostic | logs per-decode `n_kv` for the QSA path | neutral |

Supporting change: `ggml_cuda_lightning_indexer` (`lightning-indexer.cu`) extended to
the Qwen4exp QSA geometry **4 heads x 128** (`neq1 == 4`) with f16 / q4_0 / q4_1 /
q5_0 / q5_1 / q8_0 / bf16 / f32 KV types.

### B2. Sparse-attention all-`-inf` tile skip (`fattn-tile.cuh`, `fattn-vec.cuh`)

QSA top-k unmasking leaves most KV tiles fully `-inf` masked, but the upstream
`mask_to_KV_max` only trims a **trailing causal tail** — decode (Q=1) still walked all
`n_kv` tiles. This adds `fattn_tile_kv_tile_all_inf()`: peek the mask tile first
(O(width), warp+block reduce); if every entry is `-inf`, skip the K/V HBM loads
entirely (softmax-zero either way). Active in `flash_attn_tile` and
`flash_attn_ext_vec` whenever a mask is present.

Measured (L00, with the separate prefill-MMA variant): TG-60k 23.7 -> **24.8**,
TG-150k 16.1 -> **17.5**, decode GPU util drops (18/15 at 60k) — stall removed, less
work. The `fattn-mma-f16.cuh` (prefill) counterpart was **reverted**: it cost PP
(PP-60k 751 -> 481, PP-150k 644 -> 306). Skip is not O(width): 513 scattered
4-token blocks into 64-wide tiles skips ~60 % at best.

### B3. KV high-water compaction (`llama-kv-cache.cpp`)

`get_n_kv()` pads to `used_max_p1`; after `seq_rm` the holes kept decode scoring the
high-water mark (a 512-token chat behaved like a 5k indexer). Two fixes:
full slot erase now resets allocation to 0, and `QWEN4EXP_KV_COMPACT=1` packs used
cells to `[0, used)` after `seq_rm` (refuses transposed-V caches; env-gated, off).
Operationally this is also why the driver pins
`--cache-ram 0 --slot-prompt-similarity 0` — slot LCP reuse must not poison
`used_max_p1`.

### B4. PLE (per-layer token embeddings) residency

51.8 GiB embedding table gathered on CPU per token, once per forward — this arch's #1
stall class. Three local changes:

- **`LLAMA_MLOCK_PLE=1`** (`llama-model-loader.cpp`): VirtualLock-pins **only**
  `TENSOR_READ_LAZY` tables (~51 GiB PLE shard) in 256 MiB chunks, so Windows cannot
  evict them under memory pressure. Never `mmap+mlock` the whole 110 GiB GGUF (blows
  past 128 GB RAM). The loader records lazy-candidate names even when
  `--lazy-mode off`, so the pin target survives.
- Frozen placement: `--load-mode mmap --lazy-mode off` (resident). `--lazy-mode auto`
  left the table on disk: on-demand gathers cut PP to ~45 t/s.
- **CPU `GET_ROWS` threading** (`ggml-cpu.c`): gathers use the full thread pool when
  `src1 >= 32` rows **and** `src0 > 64 MiB`; decode-sized (1-row) gathers stay
  single-threaded on purpose (waking the pool costs more than the copy).

### B5. N-gram draft-table offload ("ngram offloaded")

The checkpoint's n-gram draft table is a **54.4 GiB host structure** (shard 2): lazy
mmap, gathered on CPU, never VRAM. Policy proven on this box: **speculation off** —
n-gram drafts made unique-text decode *slower* (28.8 vs 30.0 at 2k; 23.5 at 20k).
The table stays loaded and inert; `--cache-ram 0` keeps the ngram cache from
reserving RAM. Server-side addition: `POST /slots` now supports `action=erase`
**without** `--slot-save-path` (erase writes no files; save/restore still require it)
— the workflow between long chats is `POST /slots/0?action=erase`.

### B6. MoE hot-expert VRAM cache (`--moe-cache`) — new backend feature

The largest addition: an expert cache that keeps routed experts in host RAM and parks
the **hottest** ones in spare VRAM (works per GPU; explicitly supports a PCIe card
such as the P40 as a pure cache device). Not in plain llama.cpp (upstream PR #24524
family).

- New API `ggml/src/ggml-backend-moe-cache.h`; CUDA provider
  `ggml/src/ggml-cuda/moe-cache.cu/.cuh` (slabs, admission floor, LRU-ish promotion,
  async H2D hits, per-sched sessions wired through `ggml_backend_sched_*`).
- CUDA pools **trim cache slabs on allocation OOM** instead of failing
  (`ggml-cuda.cu` classic + VMM pools).
- CPU `mul_mat_id` consults cache hits and dispatches them; batched cached-expert
  matvecs run through a dedicated `mmvq` entry (`mmvq.cu`: `mul_mat_vec_q_moe_cache`,
  kept separate so base Volta kernels stay intact).
- Graph fusion: `ggml_moe_cache_can_fuse` matches up/gate/SwiGLU expert subgraphs
  (incl. `ggml_compute_forward_swiglu_masked`).
- Placement: `--moe-cache on|auto|N` (MiB/device budget) forces a **cache-aware fit**
  that parks routed experts in RAM *after* stock `--fit`, leaving free VRAM for slabs;
  enabling it disables weight repacking (`no_extra_bufts`).
- Eligibility checks (CC, host-resident experts, shape) log `requested=` /
  `resolved=`; `-lv 4` prints pools + statistics.

Status: **`--moe-cache off` on the 2x V100 Qwen driver** (all experts are already
VRAM-resident; `auto` with the P40 un-hid was killed: TG 5 t/s). The feature targets
CPU-spill hybrids — measured positive regime is the sibling 3-GPU
(2x V100+P40) Ling-3.0-flash run, where ~9.4 GiB of experts spill to RAM.

### B7. Volta decode-dispatch A/B envs (kept, both currently killed)

- `GGML_CUDA_VOLTA_DECODE_MMA=1` — force WMMA FA for small-q decode. Crashed CUDA
  init on this build (H-wave); do not enable.
- `GGML_CUDA_VOLTA_DECODE_VEC=1` — force vec FA for `Q<=8` decode (long `n_kv` /
  q8_0 KV). No TG win (H08). Documented for future geometries.

### B8. Misc local changes

- `common/arg.cpp`: `--moe-cache` parse + env `LLAMA_ARG_MOE_CACHE`; auto mode sets
  cache-aware fit; moe-cache implies `no_extra_bufts`.
- `llama-context.cpp`: cache eligibility resolution + sched wiring (main and MTP
  scheds); `llama.h`/`llama-cparams.h` expose `moe_cache_mode` /
  `moe_cache_budget_mib`.
- `ggml-backend-reg/meta`: NONE-view split fixes for the meta backend (graphs with
  host-view inputs).
- `tools/server/server-context.cpp`: slot `erase` without `--slot-save-path` (B5).

---

## Environment flag reference

| Flag | Default | Effect | Origin |
|---|---|---|---|
| `GGML_CUDA_P2P` | unset | NVLink P2P copies (upstream) | base |
| `GGML_CUDA_VOLTA_Q8_FATTN_TC` | off | q8-KV TC attention kernel | base |
| `GGML_CUDA_VOLTA_GQA8_NCOLS2` | off | exact long-K GQA8 tile geometry | base |
| `GGML_CUDA_VOLTA_Q5_X4` | off | Q5_K x4 GEMV | base |
| `GGML_CUDA_VOLTA_Q6_W4R4` | off | Q6_K warp4/row4 GEMV | base |
| `GGML_CUDA_VOLTA_FORCE_MMQ` | off | force MMQ (use for routed MoE, not dense Qwen) | base |
| `GGML_CUDA_TURING_CUBLAS_MIN_BATCH` | off | SM75 dense cuBLAS crossover | base |
| `GGML_CUDA_VOLTA_DECODE_MMA` | off | force WMMA decode FA (killed: crash) | local |
| `GGML_CUDA_VOLTA_DECODE_VEC` | off | force vec decode FA (killed: no win) | local |
| `QWEN4EXP_QSA_SLIM` | off (**driver: 1**) | block top-k + winner-only `GET_ROWS` | local |
| `QWEN4EXP_QSA_GATHER` / `_AUTO` / `_MIN_KV` | off / off / 8192 | block-gather FA path (killed) | local |
| `QWEN4EXP_QSA_POOLED_K` / `_MIN_KV` | off / 8192 | pooled indexer-K, V-scored long-ctx | local |
| `QWEN4EXP_QSA_FUSED_LID` / `_MIN_BLOCKS` | off / 2048 | fused 4-head indexer kernel | local |
| `QWEN4EXP_QSA_LOG` | off | per-decode `n_kv` logging | local |
| `QWEN4EXP_HC_PRE` | off | fused HC mean collapse (`dsv4_hc_pre`) | local |
| `QWEN4EXP_HC_COMB_BCAST` | off | broadcast HC combine (skip `repeat`) | local |
| `QWEN4EXP_KV_COMPACT` | off | pack KV cells after `seq_rm` | local |
| `LLAMA_MLOCK_PLE` | off | VirtualLock PLE tables only (chunked) | local |
| `LLAMA_ARG_MOE_CACHE` | off | `--moe-cache` via env | local |

## CLI additions

- `--moe-cache {off|auto|on|N}` — hot-expert VRAM cache, N = MiB/device budget
  (B6). Env: `LLAMA_ARG_MOE_CACHE`.
- Inherited and relied upon here: `--prefill-reuse`, `--pipeline-copies`,
  `--slot-prompt-similarity`, `--spec-mtp-defer-prompt`, `-sm layer`, `-fa on`,
  `--fit off`, `--moe-cache`, `--cache-ram 0`.

## Production launch (frozen daily driver)

Windows: see [`docs/examples/windows-dual-v100/`](docs/examples/windows-dual-v100/)
(`run-env.cmd` + `run-server.cmd` + `bench-matrix.ps1`). Equivalent one-shot
PowerShell (as run on 2026-09-06):

```powershell
$env:CUDA_VISIBLE_DEVICES = "0,1"
$env:GGML_CUDA_P2P        = "1"
$env:QWEN4EXP_QSA_SLIM    = "1"   # B1 — only QSA mode promoted
$env:LLAMA_MLOCK_PLE      = "0"   # set "1" to pin the 51 GiB PLE against page eviction

& ".\build\bin\llama-server.exe" `
  -m "...\Qwen3.8-Flash-Next-AD-5.00bpw-Q5_K_M-M64-00001-of-00033.gguf" `
  --alias qwen3.8-flash-next --jinja `
  --reasoning on --reasoning-format deepseek --reasoning-effort medium `
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 `
  -c 163840 -ngl 99 -fa on -sm layer -np 1 `
  -b 1536 -ub 768 --prefill-reuse 768 --pipeline-copies 1 `
  --cache-type-k f16 --cache-type-v f16 --fit off --moe-cache off `
  -t 18 -tb 18 --cpu-mask 0x555555555 --cpu-strict 1 --poll 1 --prio 2 `
  --load-mode mmap --lazy-mode off `
  --cache-ram 0 --slot-prompt-similarity 0 `
  --metrics --host 127.0.0.1 --port 8080
```

Why these look arbitrary — each was measured: `-sm layer` beat `-sm tensor` once the
meta/AllReduce tax was understood (TG-2048 28.1 -> 34.2); `ub 768` is the largest
ubatch that survives 163840-ctx compute buffers (1024/2048 need +2.8/+5.6 GiB and
OOM); `--fit off` (stock fit mis-sizes this arch); f16 KV (q8 cost more than it
saved); `--cache-ram 0 --slot-prompt-similarity 0` (poisoned `used_max_p1`);
physical-core mask + strict pin (`-t 18` of 36 HW threads).

## Ruled out on this box (negative results — save yourself the run)

- `-sm row` (no CUDA split buffers), `-fa off` (fails to load), `-sm tensor` ratios
  other than even, retry of P2P-off (NVLink is not the decode bottleneck at bs=1).
- q8_0 KV (TG-20k 27.3 vs f16 28.7); n-gram speculation on unique text;
  `-ncmoe 4/8`; `--moe-cache auto` with P40 visible; `QWEN4EXP_QSA_GATHER` (any
  gate); `FUSED_LID`/`POOLED_K` as always-on; `VOLTA_DECODE_MMA/_VEC`.
- `163840 + ub >= 1024` (OOM math); `96k ctx` mid-run crash with ub 2048 leftovers.
- Smaller weight quant: the long-ctx dive is QSA-stall, not weight bandwidth — Q5_K_M
  stays.

## Commit manifest

Base: `ac8d815` (upstream of this layer). Local changes to commit alongside these docs:

- Modified (27): `common/arg.cpp`, `common/common.{cpp,h}`, `include/llama.h`,
  `src/llama-{cparams,context,kv-cache,kv-cache,kv-cells,memory-hybrid-idx,model-loader,model}.cpp/.h`,
  `src/models/{models.h,qwen4exp.cpp}`, `tools/server/server-context.cpp`,
  `ggml/src/ggml-backend.{cpp,meta.cpp,reg.cpp}`, `ggml/src/ggml-cpu/ggml-cpu.c`,
  `ggml/src/ggml-cuda/{fattn.cu,fattn-tile.cuh,fattn-vec.cuh,ggml-cuda.cu,lightning-indexer.cu,mmvq.cu,mmvq.cuh}`
- New (3): `ggml/src/ggml-backend-moe-cache.h`,
  `ggml/src/ggml-cuda/moe-cache.cu`, `ggml/src/ggml-cuda/moe-cache.cuh`
- Docs (this batch): `README-FORK.md`, `docs/fork-benchmarks.md`,
  `docs/examples/windows-dual-v100/`

## Provenance

- Base fork README + measured single-GPU tables: `README.md` (§Performance,
  §Implementation), `HANDOFF_2026-09-05_SM70_SM75.md`.
- Dual-V100 sweep history (V/C/H/L waves, 100+ cells): raw JSONL in
  `docs/examples/windows-dual-v100/bench-results.jsonl`; curated in
  [`docs/fork-benchmarks.md`](docs/fork-benchmarks.md).
- Sibling project on the same binary (Ling-3.0-flash, 2x V100 + P40, bailingmoe3):
  its handoff lives outside this repo; the `--moe-cache` feature was built for its
  CPU-spill regime (see B6 status).
