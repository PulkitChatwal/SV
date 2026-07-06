import os, sys, re, json, time, statistics, subprocess, threading
MODEL = os.environ.get("SKIVE_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
N = int(os.environ.get("SKIVE_N", "100")); MAXTOK = int(os.environ.get("SKIVE_MAXTOK", "1024"))
# (mode, budget_blocks, metric, stride, block_size, sink, local)
CONFIGS = [("fullkv", None, None, 1, 16, 2, 6),
           ("evict", 32, "value_attention", 8, 16, 2, 6),    # (a) batched, stride 8
           ("evict", 4, "value_attention", 8, 128, 1, 2)]    # (a)+(b) bs128, 512-tok budget


def worker(mode, budget, metric, stride, bsz, sink, local):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    if metric: os.environ["SKIVE_METRIC"] = metric
    os.environ["SKIVE_SCORE_LAYER_STRIDE"] = str(stride); os.environ["SKIVE_SPARSE_GATHER"] = "1"
    from vllm import LLM, SamplingParams
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{N}]")
    evict = mode != "fullkv"
    kw = dict(model=MODEL, dtype="bfloat16", seed=0, enforce_eager=True, trust_remote_code=True,
              enable_prefix_caching=False, gpu_memory_utilization=0.85, max_model_len=4096,
              block_size=bsz, kv_evict_enabled=evict, kv_evict_budget=(budget if evict else 1),
              kv_evict_num_sink_blocks=sink, kv_evict_num_local_blocks=local)
    gpub = os.environ.get("GPU_BLOCKS")  # cap KV cache -> force memory-bound serving
    if gpub:
        # keep total KV bytes constant across block sizes: fewer, bigger blocks
        kw["num_gpu_blocks_override"] = max(1, int(int(gpub) * 16 // bsz))
    llm = LLM(**kw); tok = llm.get_tokenizer()
    prompts, golds = [], []
    for ex in ds:
        q = ex["question"] + "\nPlease reason step by step and put your final answer after '####'."
        prompts.append(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True))
        m = re.search(r"####\s*([-0-9,.]+)", ex["answer"]); golds.append(m.group(1).replace(",", "").rstrip(".") if m else None)
    t0 = time.perf_counter(); outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=MAXTOK)); dt = time.perf_counter() - t0
    intok = [len(o.prompt_token_ids) for o in outs]; outok = [len(o.outputs[0].token_ids) for o in outs]
    c = 0
    for o, g in zip(outs, golds):
        nums = re.findall(r"-?\d+", o.outputs[0].text.replace(",", "")); p = nums[-1] if nums else None
        try: c += int(p is not None and g is not None and abs(float(p) - float(g)) < 1e-4)
        except ValueError: c += int(p == g)
    mean_tok = statistics.mean([i + o for i, o in zip(intok, outok)])
    print("METRICS " + json.dumps(dict(mode=mode, metric=metric or "-", budget=(budget if evict else None),
        stride=stride, bsz=bsz, acc=round(100 * c / len(outs), 1), wall=round(dt, 1),
        toks=round(sum(outok) / dt, 1), outtot=sum(outok), meantok=round(mean_tok, 1))), flush=True)


def gpu_peak(stop, peak):
    while not stop.is_set():
        try:
            o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
            peak[0] = max(peak[0], int(o.strip().splitlines()[0]))
        except Exception: pass
        time.sleep(0.4)


def run(cfg):
    mode, budget, metric, stride, bsz, sink, local = cfg
    env = dict(os.environ, VLLM_USE_V2_MODEL_RUNNER="0", SKIVE_SCORE_LAYER_STRIDE=str(stride), SKIVE_SPARSE_GATHER="1")
    if metric: env["SKIVE_METRIC"] = metric
    peak, stop = [0], threading.Event(); th = threading.Thread(target=gpu_peak, args=(stop, peak)); th.start()
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", mode, str(budget), str(metric),
                        str(stride), str(bsz), str(sink), str(local)], env=env, capture_output=True, text=True)
    stop.set(); th.join(); blob = p.stdout + p.stderr; met = {}; ev = rc = 0
    for line in blob.splitlines():
        if line.startswith("METRICS "): met = json.loads(line[8:])
        m = re.search(r"\[SKIVE 4c\] evicted .*cumulative=(\d+)", line);  ev = int(m.group(1)) if m else ev
        m = re.search(r"\[SKIVE 4c-ii\] freed .*cumulative=(\d+)", line); rc = int(m.group(1)) if m else rc
    if not met: print("!! FAILED:\n" + "\n".join(blob.splitlines()[-14:])); return None
    met.update(evicts=ev, reclaim=rc, peakMB=peak[0]); return met


def main():
    rows = []
    for cfg in CONFIGS:
        lbl = cfg[0] if cfg[0] == "fullkv" else f"{cfg[2]}@{cfg[1]} s{cfg[3]} bs{cfg[4]}"
        print(f"\n===== {lbl} =====", flush=True)
        r = run(cfg)
        if r: rows.append(r)
    base = next((r for r in rows if r["mode"] == "fullkv"), None)
    bacc = base["acc"] if base else None; mt = base["meantok"] if base else None
    hdr = (f"{'config':<28}{'budget':>7}{'KVsave%':>9}{'acc%':>7}{'dAcc':>7}{'tok/s':>8}"
           f"{'wall_s':>8}{'out_tok':>9}{'evicts':>8}{'reclaim':>8}{'peakMB':>8}")
    print("\n" + "=" * len(hdr) + "\n" + hdr + "\n" + "=" * len(hdr))
    for r in rows:
        cfg = r["mode"] if r["mode"] == "fullkv" else f"{r['metric']}@{r['budget']} s{r['stride']} bs{r['bsz']}"
        kv = f"{100 * (1 - r['budget'] * r['bsz'] / mt):.0f}" if (r['budget'] and mt) else "-"
        bud = str(r['budget']) if r['budget'] else "full"
        da = f"{r['acc'] - bacc:+.1f}" if bacc is not None else "-"
        print(f"{cfg:<28}{bud:>7}{kv:>9}{r['acc']:>7}{da:>7}{r['toks']:>8}{r['wall']:>8}"
              f"{r['outtot']:>9}{r['evicts']:>8}{r['reclaim']:>8}{r['peakMB']:>8}")
    print("=" * len(hdr))
    print("KVsave% = 1 - budget_tokens/mean_seq_tokens; evicts/reclaim = cumulative blocks.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], (None if sys.argv[3] == "None" else int(sys.argv[3])),
               (None if sys.argv[4] == "None" else sys.argv[4]),
               int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]))
    else:
        main()
