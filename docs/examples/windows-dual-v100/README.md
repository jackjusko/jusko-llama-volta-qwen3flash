# Windows: Qwen3.8-Flash-Next on 2x V100 (frozen daily driver)

Production launch config + benchmark harness used for every number in
[`../../fork-benchmarks.md`](../../fork-benchmarks.md). These are live configs from
one specific machine (2x V100-SXM2 32 GB NVLink, E5-2697 v4, 128 GB RAM, CUDA 12.2,
Windows) — edit the paths/ports before use.

| File | Purpose |
|---|---|
| `run-env.cmd` | Authoritative env: GPU visibility, `GGML_CUDA_P2P`, `QWEN4EXP_QSA_SLIM=1`, `LLAMA_MLOCK_PLE`, CPU pinning, frozen GPU args (160k ctx, `-sm layer`, ub 768, F16 KV, `--fit off`). Comments explain each choice. |
| `run-server.cmd` | Wrapper: kills stale servers, adds reasoning/spec switches (`SPEC=1` attempts MTP — absent in this quant), starts `llama-server` on `:8080`. |
| `bench-matrix.ps1` | The harness: fixed workloads (PP-2k..60k, TG-64-at-N, think-512), `nvidia-smi` util sampling, one JSONL row per run. |
| `bench-results.jsonl` | Raw results for the V/C/H/L waves (100+ cells) backing the docs. |

Quick start (PowerShell, from a build of this tree):

```powershell
# 1. edit $LLAMA_BIN_DIR / $LLAMA_MODEL / ports in run-env.cmd
# 2. download the 33-shard model to $LLAMA_MODEL's folder (shard 2 = n-gram table)
cmd /c run-server.cmd

# or the equivalent explicit one-shot (see README-FORK.md "Production launch")
```

Key non-obvious flags, all measured (rationale in `README-FORK.md` and the
`run-env.cmd` comments):

- `--cache-ram 0 --slot-prompt-similarity 0` — mandatory: slot reuse otherwise
  poisons the QSA `used_max_p1` high-water mark and permanently degrades decode.
- `--fit off -ngl 99` — stock fit mis-sizes this architecture.
- `-ub 768` — largest ubatch whose compute buffers fit at 163 840 ctx.
- `--lazy-mode off` + `LLAMA_MLOCK_PLE` — keep the 51 GiB PLE table resident;
  never `mmap+mlock` the whole 110 GiB GGUF.
- Between long chats: `POST /slots/0?action=erase` (works without `--slot-save-path`
  with this tree).
