# Skive-V — vLLM + value-aware block-wise KV-cache eviction

Skive-V is a fork of **vLLM `v0.23.0`** (commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`)
that adds **SKIVE**: value-aware, block-wise KV-cache eviction. It lets you run
**more concurrent requests / longer context on the same GPU** by capping each
sequence's KV footprint — trading a controlled amount of output quality for
memory headroom.

> **One-line summary:** flag **off ⇒ byte-identical to stock vLLM**; flag on ⇒
> fewer KV blocks per sequence (≈3–4× more concurrency/context when KV-bound),
> at some quality cost. It is **not** a universal upgrade — see *When to use it*.

## How it works (no kernel changes)
- **Score** each KV block. Two metrics (`SKIVE_METRIC`):
  - `value_attention` (default, best quality) — `Σ softmax(q·k·scale) · ‖v‖₁`
    using the current decode query; keeps blocks the model actually attends to.
  - `vk_ratio` (static proxy, cheapest) — `‖V_block‖₂ / ‖K_block‖₂`, computable
    straight from the cache. Lower = evict first.
- **Evict** the lowest-importance non-protected blocks. With **sparse-gather**
  (`SKIVE_SPARSE_GATHER=1`, default) the evicted blocks are dropped from the
  FlashAttention read by compacting the block table and shortening `seqused_k`,
  and the physical blocks are freed — RoPE-exact, causal-safe, decode-only. The
  first `num_sink_blocks` and last `num_local_blocks` are always protected.
- **Fast, low-overhead scoring:** all over-budget requests are scored in one
  fused GPU op with a single CPU sync per step (`_batch_score_va`); scoring can
  be sub-sampled to every k-th attention layer (`SKIVE_SCORE_LAYER_STRIDE=8`,
  ~k× cheaper); eviction can use large blocks (`block_size=128`) for ~7× fewer
  passes.
- All changes are pure Python in `vllm/` + a vendored `vllm/kv_evict/` package.

**Full results, formulas, and methodology:** see
[`SKIVE_RESULTS.md`](SKIVE_RESULTS.md) and reproduction scripts in
[`benchmarks/skive/`](benchmarks/skive/) (LongBench, GSM8K, throughput sweeps).
Headline: on LongBench (R1-Distill-1.5B, no cache cap) up to **82% KV saved,
+39% throughput**, near-lossless accuracy; under memory pressure **2.15× FullKV**.

## Build (from source)
SKIVE is pure Python, but this is a full vLLM source tree, so it builds like
upstream vLLM. On a CUDA box:

```bash
git clone https://github.com/PulkitChatwal/Skive-V.git && cd Skive-V
pip install cmake ninja setuptools-rust          # + a Rust toolchain (cargo)
python use_existing_torch.py                      # build against your torch
MAX_JOBS=6 TORCH_CUDA_ARCH_LIST="8.9" \
    pip install -e . --no-build-isolation         # set arch to YOUR GPU (8.9 = L4)
```
(Build time is dominated by CUDA kernel compilation; limiting `TORCH_CUDA_ARCH_LIST`
to your GPU's arch keeps it to ~1 hr.)

## Usage
**Requirement:** eviction hooks live in vLLM's **V1 model runner**, so always set
`VLLM_USE_V2_MODEL_RUNNER=0` (some architectures, e.g. Llama, default to V2 —
the engine will *raise loudly* if eviction is enabled on V2).

```python
import os
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    enable_prefix_caching=False,        # supported; see notes
    kv_evict_enabled=True,              # default False == stock vLLM
    kv_evict_budget=16,                 # max KV blocks kept per sequence
    kv_evict_num_sink_blocks=2,         # leading blocks never evicted
    kv_evict_num_local_blocks=4,        # trailing (recent) blocks never evicted
)
print(llm.generate(["Hello"], SamplingParams(max_tokens=64))[0].outputs[0].text)
```
CLI equivalents: `--kv-evict-enabled --kv-evict-budget 16 --kv-evict-num-sink-blocks 2 --kv-evict-num-local-blocks 4`.

## When to use it
- ✅ **You are KV-bound** (long context / high concurrency filling the GPU) and can
  tolerate some quality loss → fits ~3–4× more on the same VRAM; measured
  **+24% to +60% throughput** in a KV-bound benchmark.
- ❌ **You are not KV-bound** → eviction only adds per-step overhead (measured
  **−8% to −16%**); just leave it off (or don't use this fork).
- ⚠️ **Quality is lossy** and worst at the aggressive budgets that save the most
  memory (token-overlap ≈0.78 vs FullKV at ~80% budget on a 0.5B model; coherent,
  never gibberish). Validate on *your* task before relying on it.

## Status / caveats
- Pinned to vLLM `0.23.0`. The eviction patches are version-specific.
- Validated on Qwen2.5-0.5B and TinyLlama-1.1B (GQA + a second arch).
- Prefix caching is supported (shared blocks are ref-counted, not corrupted).
- Preemption: eviction reduces the memory pressure that causes it.
- Quality numbers above are token-divergence vs FullKV, **not** a task-accuracy
  eval — run a real eval for your workload.

## Attribution & license
Based on [vLLM](https://github.com/vllm-project/vllm) `v0.23.0`, Apache-2.0.
SKIVE modifications by PulkitChatwal. See `LICENSE` (Apache-2.0, retained from
vLLM) and `NOTICE`.
