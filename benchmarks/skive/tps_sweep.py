import os, sys, re, json, time, subprocess, threading
# Fixed capped KV memory. Sweep offered batch upward. FullKV saturates its cache
# and plateaus (preemption/queueing); SKIVE's smaller footprint keeps scaling.
MODEL = os.environ.get("SKIVE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
REP = int(os.environ.get("CTX_REPEAT", "45"))       # prompt length knob
MAXTOK = int(os.environ.get("MAXTOK", "256"))
GPUB = int(os.environ.get("GPU_BLOCKS", "5000"))    # the fixed KV cap (bs16 blocks)
BATCHES = [int(x) for x in os.environ.get("BATCHES", "64,128,256,512").split(",")]
# (label, mode, metric, stride, budget)
CONFIGS = [("FullKV", "off", None, 1, None),
           ("SKIVE va@16 s8", "evict", "value_attention", 8, 16)]


def worker(label, mode, metric, stride, budget, batch):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    if metric: os.environ["SKIVE_METRIC"] = metric
    os.environ["SKIVE_SCORE_LAYER_STRIDE"] = str(stride); os.environ["SKIVE_SPARSE_GATHER"] = "1"
    from vllm import LLM, SamplingParams
    kw = dict(model=MODEL, dtype="bfloat16", enforce_eager=True, enable_prefix_caching=False,
              gpu_memory_utilization=0.85, max_model_len=4096, seed=0,
              num_gpu_blocks_override=GPUB)
    if mode != "off":
        kw.update(kv_evict_enabled=True, kv_evict_budget=budget,
                  kv_evict_num_sink_blocks=2, kv_evict_num_local_blocks=6)
    llm = LLM(**kw)
    base = ("In a distant land there lived many creatures who studied the stars "
            "and recorded their motions carefully over many generations. ")
    prompt = base * REP + "\n\nContinue the story in great detail:"
    prompts = [prompt] * batch
    sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK, ignore_eos=True)
    llm.generate(prompts[:4], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))
    t0 = time.perf_counter(); outs = llm.generate(prompts, sp); dt = time.perf_counter() - t0
    tot = sum(len(o.outputs[0].token_ids) for o in outs)
    print("TP " + json.dumps(dict(label=label, batch=batch, out_tot=tot,
          wall=round(dt, 2), tok_s=round(tot / dt, 1))), flush=True)


def gpu_peak(stop, peak):
    while not stop.is_set():
        try:
            o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                "--format=csv,noheader,nounits"], text=True)
            peak[0] = max(peak[0], int(o.strip().splitlines()[0]))
        except Exception: pass
        time.sleep(0.4)


def run(label, mode, metric, stride, budget, batch):
    env = dict(os.environ)
    peak, stop = [0], threading.Event(); th = threading.Thread(target=gpu_peak, args=(stop, peak)); th.start()
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", label, mode,
                        str(metric), str(stride), str(budget), str(batch)], env=env, capture_output=True, text=True)
    stop.set(); th.join(); blob = p.stdout + p.stderr; met = None; preempt = 0
    for line in blob.splitlines():
        if line.startswith("TP "): met = json.loads(line[3:])
        if "reempt" in line:
            m = re.search(r"(\d+)\s+req", line);  preempt = max(preempt, int(m.group(1)) if m else preempt + 1)
    if not met: print(f"  !! {label} b{batch} FAILED:\n   " + "\n   ".join(blob.splitlines()[-8:])); return None
    met["preempt"] = preempt; met["peakMB"] = peak[0]; return met


def main():
    print(f"MODEL={MODEL}  KV_CAP={GPUB} blocks  ctx~{REP*18} tok  maxtok={MAXTOK}\n", flush=True)
    grid = {}
    for batch in BATCHES:
        for label, mode, metric, stride, budget in CONFIGS:
            print(f"--- {label}  batch={batch} ---", flush=True)
            r = run(label, mode, metric, stride, budget, batch)
            if r: grid[(label, batch)] = r
    hdr = f"{'batch':>7} | " + " | ".join(f"{c[0]:>16}" for c in CONFIGS) + " |  speedup"
    print("\n" + "=" * len(hdr) + "\n" + hdr + "\n" + "=" * len(hdr))
    for batch in BATCHES:
        cells = []
        for label, *_ in CONFIGS:
            r = grid.get((label, batch))
            cells.append(f"{r['tok_s']:>8.0f} t/s" if r else f"{'--':>12}")
        f = grid.get((CONFIGS[0][0], batch)); s = grid.get((CONFIGS[1][0], batch))
        sp = f"{s['tok_s']/f['tok_s']:.2f}x" if (f and s and f['tok_s']) else "-"
        print(f"{batch:>7} | " + " | ".join(f"{c:>16}" for c in cells) + f" |  {sp:>7}")
    print("=" * len(hdr))
    print("\nDetail (wall_s / out_tok / preemptions / peakMB):")
    for batch in BATCHES:
        for label, *_ in CONFIGS:
            r = grid.get((label, batch))
            if r: print(f"  b{batch:<4} {label:<16} {r['tok_s']:>7.0f} t/s  "
                        f"wall={r['wall']:>6}s  out={r['out_tot']:>7}  preempt={r['preempt']:>3}  peak={r['peakMB']}MB")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3], (None if sys.argv[4] == "None" else sys.argv[4]),
               int(sys.argv[5]), (None if sys.argv[6] == "None" else int(sys.argv[6])), int(sys.argv[7]))
    else:
        main()
