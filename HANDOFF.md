# Skive-V — run the LongBench benchmark (hand-off guide)

This reproduces the **SKIVE vs FullKV** KV-cache-eviction results on **LongBench**
(long-context QA/summarization) using **`microsoft/Phi-3.5-mini-instruct`**.
SKIVE is a value-aware, block-wise KV-cache eviction integrated into vLLM 0.23.0.

**No source build required** — SKIVE is pure Python, so we install the prebuilt
vLLM wheel and overlay SKIVE onto it.

---

## Requirements
- A **CUDA GPU**, ≥ 16 GB (24 GB L4 / A10 / A100 ideal). *CPU-only will not work.*
- Linux, Python 3.10–3.12, NVIDIA driver + CUDA (the vLLM wheel pulls a matching torch).
- Internet access (downloads the vLLM wheel and the Phi model from Hugging Face).

## Run it (copy-paste)

```bash
# 1. fresh environment (avoids dependency clashes) — pick ONE:
python -m venv skive-env && source skive-env/bin/activate
#   or:  conda create -y -n skive python=3.11 && conda activate skive

# 2. clone and run
git clone https://github.com/PulkitChatwal/SV.git skive
cd skive
bash run_longbench.sh
```

The script prints a preflight (GPU check), installs vLLM 0.23.0 (~2 min), overlays
SKIVE, then runs the benchmark. Total time depends on the GPU (roughly 15–40 min
for the 4 tasks). Output is printed and saved to:
- `longbench_results.txt` — full log
- `longbench_results.md` — just the result tables (paste into a report)

## What it produces

For each of 4 LongBench tasks, a table comparing **FullKV** vs
**`value_attention s8 bs128`** (SKIVE):

```
===== LongBench: hotpotqa  (f1) =====
config                      budget  KVsave%   acc%   dAcc   tok/s  wall_s  out_tok  evicts reclaim  peakMB
fullkv                        full        -    ...    +0.0   ...     ...      ...        0       0     ...
value_attention s8 bs128        32      ~70    ...    ...    ...     ...      ...      ...     ...     ...
```

Columns: `budget` = KV blocks kept/seq · `KVsave%` = % of KV cache dropped ·
`acc%` = task score (F1 / ROUGE) · `dAcc` = vs FullKV · `tok/s` = throughput ·
`evicts`/`reclaim` = cumulative blocks evicted/freed · `peakMB` = peak GPU MB.

**What to look for:** SKIVE drops a large fraction of the KV cache (`KVsave%`) with
a small accuracy change (`dAcc`), and — because long contexts make the run
KV-bound — a **throughput gain** (higher `tok/s`, lower `wall_s`).

## Tuning (environment variables)

```bash
NPER=3 TASKS=hotpotqa bash run_longbench.sh          # quick smoke test first
SKIVE_MODEL="microsoft/Phi-4-mini-instruct" bash run_longbench.sh   # other Phi model
NPER=40 bash run_longbench.sh                         # more examples (needs more VRAM)
NPER=10 MAXLEN=16384 GPUMEM=0.92 bash run_longbench.sh   # if out-of-memory
```

| var | default | meaning |
|---|---|---|
| `SKIVE_MODEL` | `microsoft/Phi-3.5-mini-instruct` | HF model id |
| `NPER` | `20` | examples per task (higher = more concurrency + VRAM) |
| `TASKS` | `hotpotqa,musique,narrativeqa,gov_report` | LongBench tasks |
| `MAXLEN` | `20480` | max context length |
| `GPUMEM` | `0.90` | GPU memory fraction |

## Notes / expectations
- **Phi-3.5-mini uses MHA** (heavier KV cache than GQA models), so the run is
  naturally **memory-bound** — this is where SKIVE's throughput advantage shows.
  If FullKV is slow or OOMs at high `NPER`, that is the memory pressure SKIVE relieves.
- To turn eviction **off** and get stock vLLM behavior, the `fullkv` row already
  does that (SKIVE is inert unless `kv_evict_enabled=True`).
- Details, formulas, and larger result tables: [`SKIVE_RESULTS.md`](SKIVE_RESULTS.md).

## What's under the hood (1 paragraph)
SKIVE = a ~1,000-line eviction package (`vllm/kv_evict/`) plus ~120 lines of hooks
inserted into 6 vLLM files (config, CLI, V1 model runner, FlashAttention backend,
scheduler). It scores each KV block (`value_attention`: attention-weighted value
norm), evicts the lowest-value non-protected blocks, and **physically drops them
from the FlashAttention read** (sparse-gather) so freed memory raises concurrency.
Flag off ⇒ byte-identical to stock vLLM.
