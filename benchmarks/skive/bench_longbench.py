import os, sys, re, json, time, string, zipfile, subprocess, threading
from collections import Counter
# LongBench v1, NO cache capping. Long contexts (9k-18k) make KV the bottleneck
# naturally, so eviction's throughput win shows without num_gpu_blocks_override.
MODEL = os.environ.get("SKIVE_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
NPER = int(os.environ.get("NPER", "40"))          # examples per task
TASKS = os.environ.get("TASKS", "narrativeqa,hotpotqa,musique,gov_report").split(",")
# (label, mode, metric, stride, block_size, budget, sink, local)
CONFIGS = [("fullkv", "off", None, 1, 16, None, 2, 6),
           ("value_attention s8 bs128", "evict", "value_attention", 8, 128, 32, 1, 2)]

# official LongBench per-task prompt + max generation length + metric family
PROMPT = {
 "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
 "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
 "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
 "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
}
MAXGEN = {"narrativeqa": 128, "hotpotqa": 32, "musique": 32, "gov_report": 512}
METRIC = {"narrativeqa": "f1", "hotpotqa": "f1", "musique": "f1", "gov_report": "rouge"}


def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, gts):
    best = 0.0
    for gt in gts:
        pt, gtt = norm(pred).split(), norm(gt).split()
        common = Counter(pt) & Counter(gtt); ns = sum(common.values())
        if ns == 0: continue
        p = ns / len(pt); r = ns / len(gtt); best = max(best, 2 * p * r / (p + r))
    return best


def rouge_l(pred, gts):
    def lcs(a, b):
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]
    best = 0.0
    for gt in gts:
        pt, gtt = norm(pred).split(), norm(gt).split()
        if not pt or not gtt: continue
        l = lcs(pt, gtt); p = l / len(pt); r = l / len(gtt)
        if p + r > 0: best = max(best, 2 * p * r / (p + r))
    return best


def load_task(task):
    from huggingface_hub import hf_hub_download
    zp = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    rows = []
    with zipfile.ZipFile(zp) as z:
        with z.open(f"data/{task}.jsonl") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def worker(label, mode, metric, stride, bsz, budget, sink, local, task):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    if metric: os.environ["SKIVE_METRIC"] = metric
    os.environ["SKIVE_SCORE_LAYER_STRIDE"] = str(stride); os.environ["SKIVE_SPARSE_GATHER"] = "1"
    from vllm import LLM, SamplingParams
    rows = load_task(task)[:NPER]
    evict = mode != "off"
    kw = dict(model=MODEL, dtype="bfloat16", seed=0, enforce_eager=True, trust_remote_code=True,
              enable_prefix_caching=False, gpu_memory_utilization=0.90, max_model_len=32768,
              block_size=bsz, kv_evict_enabled=evict, kv_evict_budget=(budget if evict else 1),
              kv_evict_num_sink_blocks=sink, kv_evict_num_local_blocks=local)
    llm = LLM(**kw); tok = llm.get_tokenizer()
    prompts, golds = [], []
    for r in rows:
        p = PROMPT[task].format(context=r["context"], input=r["input"])
        ids = tok(p, truncation=False)["input_ids"]
        if len(ids) > 31500:  # middle-truncate like official LongBench
            half = 15750; ids = ids[:half] + ids[-half:]
            p = tok.decode(ids, skip_special_tokens=True)
        prompts.append(p); golds.append(r["answers"])
    sp = SamplingParams(temperature=0.0, max_tokens=MAXGEN[task])
    t0 = time.perf_counter(); outs = llm.generate(prompts, sp); dt = time.perf_counter() - t0
    scorer = f1_score if METRIC[task] == "f1" else rouge_l
    sc = sum(scorer(o.outputs[0].text.strip(), g) for o, g in zip(outs, golds)) / len(outs)
    intok = sum(len(o.prompt_token_ids) for o in outs); outok = sum(len(o.outputs[0].token_ids) for o in outs)
    print("METRICS " + json.dumps(dict(label=label, task=task, budget=(budget if evict else None), bsz=bsz,
        score=round(100 * sc, 1), wall=round(dt, 1), toks=round(outok / dt, 1),
        outtot=outok, intot=intok, n=len(outs))), flush=True)


def gpu_peak(stop, peak):
    while not stop.is_set():
        try:
            o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                "--format=csv,noheader,nounits"], text=True)
            peak[0] = max(peak[0], int(o.strip().splitlines()[0]))
        except Exception: pass
        time.sleep(0.4)


def run(cfg, task):
    label, mode, metric, stride, bsz, budget, sink, local = cfg
    env = dict(os.environ)
    peak, stop = [0], threading.Event(); th = threading.Thread(target=gpu_peak, args=(stop, peak)); th.start()
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", label, mode, str(metric),
        str(stride), str(bsz), str(budget), str(sink), str(local), task], env=env, capture_output=True, text=True)
    stop.set(); th.join(); blob = p.stdout + p.stderr; met = {}; ev = rc = 0
    for line in blob.splitlines():
        if line.startswith("METRICS "): met = json.loads(line[8:])
        m = re.search(r"\[SKIVE 4c\] evicted .*cumulative=(\d+)", line);  ev = int(m.group(1)) if m else ev
        m = re.search(r"\[SKIVE 4c-ii\] freed .*cumulative=(\d+)", line); rc = int(m.group(1)) if m else rc
    if not met: print(f"  !! {label}/{task} FAILED:\n   " + "\n   ".join(blob.splitlines()[-10:])); return None
    met.update(evicts=ev, reclaim=rc, peakMB=peak[0]); return met


def main():
    print(f"MODEL={MODEL}  tasks={TASKS}  n/task={NPER}  NO cache cap\n", flush=True)
    allres = {}
    for task in TASKS:
        print(f"\n########## {task} ##########", flush=True)
        for cfg in CONFIGS:
            print(f"--- {cfg[0]} / {task} ---", flush=True)
            r = run(cfg, task)
            if r: allres[(cfg[0], task)] = r
    hdr = (f"{'config':<27}{'budget':>7}{'KVsave%':>9}{'acc%':>7}{'dAcc':>7}{'tok/s':>8}"
           f"{'wall_s':>8}{'out_tok':>9}{'evicts':>8}{'reclaim':>8}{'peakMB':>8}")
    for task in TASKS:
        print(f"\n===== LongBench: {task}  ({METRIC[task]}) =====")
        print("=" * len(hdr) + "\n" + hdr + "\n" + "=" * len(hdr))
        base = allres.get((CONFIGS[0][0], task)); bacc = base["score"] if base else None
        mean_tok = (base["intot"] + base["outtot"]) / base["n"] if base else None
        for cfg in CONFIGS:
            r = allres.get((cfg[0], task))
            if not r: continue
            kv = f"{100 * (1 - r['budget'] * r['bsz'] / mean_tok):.0f}" if (r['budget'] and mean_tok) else "-"
            bud = str(r['budget']) if r['budget'] else "full"
            da = f"{r['score'] - bacc:+.1f}" if (bacc is not None and r['budget']) else "+0.0"
            print(f"{cfg[0]:<27}{bud:>7}{kv:>9}{r['score']:>7}{da:>7}{r['toks']:>8}{r['wall']:>8}"
                  f"{r['outtot']:>9}{r['evicts']:>8}{r['reclaim']:>8}{r['peakMB']:>8}")
        print("=" * len(hdr))
    print("\nKVsave% = 1 - budget*block_size / mean_seq_tokens.  No num_gpu_blocks_override.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3], (None if sys.argv[4] == "None" else sys.argv[4]),
               int(sys.argv[5]), int(sys.argv[6]), (None if sys.argv[7] == "None" else int(sys.argv[7])),
               int(sys.argv[8]), int(sys.argv[9]), sys.argv[10])
    else:
        main()
