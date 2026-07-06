import os, sys, time, json
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
mode = sys.argv[1]; budget = int(sys.argv[2]) if len(sys.argv) > 2 else 16
REP = int(os.environ.get("CTX_REPEAT", "45")); BATCH = int(os.environ.get("BATCH", "128"))
MAXTOK = int(os.environ.get("MAXTOK", "200")); GPUB = os.environ.get("GPU_BLOCKS")
from vllm import LLM, SamplingParams
kw = dict(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="bfloat16", enforce_eager=True,
          enable_prefix_caching=False, gpu_memory_utilization=0.85, max_model_len=4096, seed=0)
if GPUB:
    kw["num_gpu_blocks_override"] = int(GPUB)
if mode == "off":
    kw["kv_evict_enabled"] = False
else:
    kw.update(kv_evict_enabled=True, kv_evict_budget=budget,
              kv_evict_num_sink_blocks=2, kv_evict_num_local_blocks=6)
llm = LLM(**kw)
base = ("In a distant land there lived many creatures who studied the stars "
        "and recorded their motions carefully over many generations. ")
prompt = base * REP + "\n\nContinue the story in great detail:"
prompts = [prompt] * BATCH
sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK, ignore_eos=True)
llm.generate(prompts[:4], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))
t0 = time.perf_counter(); outs = llm.generate(prompts, sp); dt = time.perf_counter() - t0
tot = sum(len(o.outputs[0].token_ids) for o in outs)
print("TP", json.dumps({"mode": mode, "budget": budget, "batch": BATCH,
      "metric": os.environ.get("SKIVE_METRIC", "vk_ratio"),
      "stride": int(os.environ.get("SKIVE_SCORE_LAYER_STRIDE", "1")),
      "out_tot": tot, "wall_s": round(dt, 2), "tok_s": round(tot / dt, 1)}))
