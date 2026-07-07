# Quickstart — reproduce the LongBench table on a Phi model

**No source build needed.** SKIVE is pure Python, so this installs the *prebuilt*
vLLM 0.23.0 wheel and overlays SKIVE on top. On any CUDA GPU box:

```bash
git clone https://github.com/PulkitChatwal/SV.git skive
cd skive
bash run_longbench.sh
```

That's it. It will:
1. `pip install vllm==0.23.0` (prebuilt, ~2 min) + `datasets huggingface_hub`
2. Overlay SKIVE's Python files onto the installed vLLM
3. Run FullKV vs `value_attention s8 bs128` on 4 LongBench tasks and print the table
   (also saved to `longbench_results.txt`)

Expected output (numbers vary by GPU/model):

```
===== LongBench: hotpotqa  (f1) =====
config                      budget  KVsave%   acc%   dAcc   tok/s  wall_s  out_tok  evicts reclaim  peakMB
fullkv                        full        -   ...    +0.0   ...     ...     ...        0       0    ...
value_attention s8 bs128        32      ~70   ...    ...    ...     ...     ...      ...     ...    ...
```

## Requirements
- A **CUDA GPU** (≥16 GB; 24 GB L4/A10 recommended). CPU-only won't work.
- Python 3.10–3.12, a working PyTorch/CUDA (the vLLM wheel pulls a compatible torch).
- Use a **fresh** virtualenv or conda env to avoid dependency clashes.

## Choosing the model / tuning

Override any of these via environment variables:

```bash
# a different Phi model
SKIVE_MODEL="microsoft/Phi-4-mini-instruct" bash run_longbench.sh

# more examples per task (needs more VRAM), all 4 tasks
NPER=40 bash run_longbench.sh

# if you hit OOM on a smaller GPU, shrink the batch / context
NPER=10 MAXLEN=16384 GPUMEM=0.92 bash run_longbench.sh

# a single task, quick smoke test
NPER=3 TASKS=hotpotqa bash run_longbench.sh
```

| var | default | meaning |
|---|---|---|
| `SKIVE_MODEL` | `microsoft/Phi-3.5-mini-instruct` | HF model id |
| `NPER` | `20` | examples per task (concurrency) |
| `TASKS` | `hotpotqa,musique,narrativeqa,gov_report` | LongBench tasks |
| `MAXLEN` | `20480` | max context length |
| `GPUMEM` | `0.90` | GPU memory fraction |

## What the columns mean
`budget` = KV blocks kept/seq · `KVsave%` = fraction of KV cache dropped ·
`acc%` = task score (F1 or ROUGE) · `dAcc` = vs FullKV · `tok/s` = throughput ·
`evicts`/`reclaim` = cumulative blocks evicted/freed · `peakMB` = peak GPU memory.

Full methodology and larger result tables: [`SKIVE_RESULTS.md`](SKIVE_RESULTS.md).
