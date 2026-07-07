#!/usr/bin/env bash
# ==========================================================================
# Skive-V — one-command LongBench benchmark (FullKV vs SKIVE) on a Phi model.
#
#   Clone this repo on a CUDA GPU box, then:   bash run_longbench.sh
#
# It installs the PREBUILT vLLM 0.23.0 wheel (no source build, ~2 min),
# overlays SKIVE's pure-Python changes onto it, and runs the benchmark.
# Results are printed and saved to longbench_results.txt.
# ==========================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---- config (override via env) ------------------------------------------
export SKIVE_MODEL="${SKIVE_MODEL:-microsoft/Phi-3.5-mini-instruct}"
export NPER="${NPER:-20}"                                   # examples per task
export TASKS="${TASKS:-hotpotqa,musique,narrativeqa,gov_report}"
export MAXLEN="${MAXLEN:-20480}"                            # ctx cap (LongBench <=18k)
export GPUMEM="${GPUMEM:-0.90}"                             # GPU mem fraction
export VLLM_USE_V2_MODEL_RUNNER=0                           # SKIVE lives in V1 runner
export SKIVE_SPARSE_GATHER=1                                # sparse-gather eviction

echo "== 1/3 installing vLLM 0.23.0 (prebuilt wheel) + benchmark deps =="
python -m pip install -q "vllm==0.23.0" "datasets" "huggingface_hub" "numpy"

echo "== 2/3 overlaying SKIVE (pure-Python) onto the installed vLLM =="
VLLM_DIR="$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
for f in config/cache.py engine/arg_utils.py v1/worker/gpu_model_runner.py \
         v1/attention/backends/flash_attn.py v1/engine/core.py v1/worker/gpu/model_runner.py; do
  cp "vllm/$f" "$VLLM_DIR/$f"
done
rm -rf "$VLLM_DIR/kv_evict"
cp -r vllm/kv_evict "$VLLM_DIR/kv_evict"
echo "   overlaid into: $VLLM_DIR"
python -c "import vllm.kv_evict.integration as m; print('   SKIVE import OK:', hasattr(m, '_batch_score_va'))"

echo "== 3/3 running LongBench: $SKIVE_MODEL  |  tasks=$TASKS  |  n/task=$NPER =="
python benchmarks/skive/bench_longbench.py 2>&1 | tee longbench_results.txt

echo ""
echo "== done. table saved to: $(pwd)/longbench_results.txt =="
