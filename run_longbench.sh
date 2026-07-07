#!/usr/bin/env bash
# ==========================================================================
# Skive-V — one-command LongBench benchmark (FullKV vs SKIVE) on a Phi model.
#
#   On a CUDA GPU box:
#       git clone https://github.com/PulkitChatwal/SV.git skive
#       cd skive
#       bash run_longbench.sh
#
# SKIVE is pure Python, so this installs the PREBUILT vLLM 0.23.0 wheel
# (no source build, ~2 min), overlays SKIVE's Python onto it, and runs the
# benchmark. Table is printed and saved to longbench_results.txt / .md.
# ==========================================================================
set -euo pipefail
cd "$(dirname "$0")"
trap 'echo; echo "!! FAILED at line $LINENO. See the message above."; echo "   Common fixes: use a fresh venv/conda env; lower NPER/MAXLEN if OOM."; exit 1' ERR

# ---- config (override via env) ------------------------------------------
export SKIVE_MODEL="${SKIVE_MODEL:-microsoft/Phi-3.5-mini-instruct}"
export NPER="${NPER:-20}"                                   # examples per task
export TASKS="${TASKS:-hotpotqa,musique,narrativeqa,gov_report}"
export MAXLEN="${MAXLEN:-20480}"                            # ctx cap (LongBench <=18k)
export GPUMEM="${GPUMEM:-0.90}"                             # GPU mem fraction
export VLLM_USE_V2_MODEL_RUNNER=0                           # SKIVE lives in the V1 runner
export SKIVE_SPARSE_GATHER=1                                # sparse-gather eviction

echo "=========================================================="
echo " Skive-V LongBench runner"
echo "   model   : $SKIVE_MODEL"
echo "   tasks   : $TASKS   (n/task=$NPER)"
echo "   ctx cap : $MAXLEN   gpu_mem: $GPUMEM"
echo "=========================================================="

# ---- 0/3 preflight -------------------------------------------------------
echo "== 0/3 preflight =="
command -v python >/dev/null || { echo "!! no 'python' on PATH"; exit 1; }
python -c 'import sys; assert sys.version_info[:2] >= (3,10), "need Python >=3.10"; print("   python", sys.version.split()[0])'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | sed 's/^/   GPU: /'
else
  echo "!! nvidia-smi not found — this benchmark REQUIRES a CUDA GPU. Aborting."; exit 1
fi

# ---- 1/3 install ---------------------------------------------------------
echo "== 1/3 installing vLLM 0.23.0 (prebuilt wheel) + benchmark deps =="
python -m pip install -q "vllm==0.23.0" "datasets" "huggingface_hub" "numpy"

# ---- 2/3 overlay SKIVE (pure-Python) onto the installed vLLM -------------
echo "== 2/3 overlaying SKIVE onto the installed vLLM =="
VLLM_DIR="$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
for f in config/cache.py engine/arg_utils.py v1/worker/gpu_model_runner.py \
         v1/attention/backends/flash_attn.py v1/engine/core.py v1/worker/gpu/model_runner.py; do
  test -f "vllm/$f" || { echo "!! missing source vllm/$f in repo"; exit 1; }
  test -f "$VLLM_DIR/$f" || { echo "!! installed vLLM missing $f — version mismatch?"; exit 1; }
  cp "vllm/$f" "$VLLM_DIR/$f"
done
rm -rf "$VLLM_DIR/kv_evict"
cp -r vllm/kv_evict "$VLLM_DIR/kv_evict"
python -c "import vllm.kv_evict.integration as m; assert hasattr(m,'_batch_score_va'); print('   overlay OK ->', '$VLLM_DIR')"

# ---- 3/3 run -------------------------------------------------------------
echo "== 3/3 running LongBench (this loads the model per config; be patient) =="
python benchmarks/skive/bench_longbench.py 2>&1 | tee longbench_results.txt

# save a markdown copy of just the tables (easy to paste into a report)
awk '/^===== LongBench/{p=1} p' longbench_results.txt > longbench_results.md 2>/dev/null || true

echo
echo "== done =="
echo "   full log : $(pwd)/longbench_results.txt"
echo "   tables   : $(pwd)/longbench_results.md"
