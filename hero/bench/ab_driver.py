"""HERO A/B driver: one engine load per (repetition, arm), arms in a shuffled order each repetition, the
same greedy prompts in every load.  Writes results.jsonl (one row per request) and the engine log of every load.

usage: python ab_driver.py BASE_CONFIG EXE OUT_DIR [--reps N] [--max-tokens N] [--arms a,b,...] [--expert-cache N]
"""
import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# arm -> (extra engine args, extra env).  HERO is the default; HERO=0 turns it off.  --pcie-frac disables both.
ARMS = {
    "cpu": (["--pcie-frac", "0"], {}),            # every miss on the CPU (Fiddler-like)
    "gpu": (["--pcie-frac", "1"], {}),            # every miss over PCIe (GPU offloading)
    "fixed": ([], {"HERO": "0"}),                 # the engine's default share (probe + PcieAuto)
    "balance": ([], {"HERO": "0", "HERO_PCIE_BALANCE": "1"}),
    "hero": ([], {"HERO": "1"}),                  # explicit; the engine's default anyway
}
PROMPTS = {
    "code": "Write a Python function that merges overlapping intervals in a list and explain its time complexity.",
    "math": "A train leaves at 9:40 and travels 312 km at 78 km/h. A second train leaves the same station at 10:15 at "
            "96 km/h on the same track. When and where does the second train catch up? Show each step.",
    "prose": "Write a short story of about 300 words about a lighthouse keeper who finds a message in a bottle.",
    "zh": "请用中文解释什么是混合专家模型(Mixture-of-Experts),以及它为什么可以在增加参数量的同时控制计算量。",
    "facts": "List ten important events in the history of the printing press, with a one-sentence explanation each.",
    "summary": "Summarize the trade-offs between caching, prefetching and CPU execution when serving large "
               "mixture-of-experts language models on a single consumer GPU with limited memory.",
}
WARMUP = "Say hello in five languages."
PORT = 8095


def drop(args, flag, with_value):
    args = list(args)
    while flag in args:
        i = args.index(flag)
        del args[i:i + (2 if with_value else 1)]
    return args


def post(path, body, timeout):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ready(timeout_s, srv):
    t0 = time.time()
    while time.time() - t0 < timeout_s and srv.poll() is None:   # a server that exited is never ready
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                h = json.loads(r.read() or b"{}")
                if r.status == 200 and h.get("status") in ("ok", "ready", None):
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def ask(text, max_tokens):
    body = {"model": "hero", "messages": [{"role": "user", "content": text}], "temperature": 0,
            "max_tokens": max_tokens, "stream": False, "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none"}
    t0 = time.time()
    r = post("/v1/chat/completions", body, timeout=1800)
    content = (r.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    tim = r.get("timings") or {}
    return {"wall_s": round(time.time() - t0, 3), "decode_tok_s": tim.get("predicted_per_second"),
            "prompt_tok_s": tim.get("prompt_per_second"), "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
            "sha": hashlib.sha256(content.encode()).hexdigest()[:16]}


def run_load(base, exe, arm, rep, out, a):
    extra, env_extra = ARMS[arm]
    cfg = json.loads(json.dumps(base))
    args = drop(drop(cfg["args"], "--vision", False), "--pcie-frac", True)
    if a.expert_cache:
        args = drop(args, "--expert-cache", True) + ["--expert-cache", str(a.expert_cache)]
    cfg.update(exe=exe, args=args + extra, env={**(cfg.get("env") or {}), **env_extra}, port=PORT)
    cfg.pop("vision", None)
    tag = f"r{rep}-{arm}"
    cfg["log"] = str(out / f"{tag}.engine.log")
    cpath = out / f"{tag}.json"
    cpath.write_text(json.dumps(cfg, indent=1))
    log = open(out / f"{tag}.server.log", "w")
    srv = subprocess.Popen([a.python, "-m", "serve.server", "--engine", "hero", "--config", str(cpath),
                            "--port", str(PORT)], cwd=a.src, stdout=log, stderr=subprocess.STDOUT,
                           start_new_session=True)
    rows = []
    try:
        if not ready(1800, srv):
            print(f"{tag}: not ready", flush=True)
            return rows
        ask(WARMUP, 64)
        for name, text in PROMPTS.items():
            row = {"rep": rep, "arm": arm, "prompt": name, **ask(text, a.max_tokens)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        try:
            os.killpg(srv.pid, signal.SIGTERM)
            srv.wait(120)
        except ProcessLookupError:   # the server had already exited (its log says why)
            pass
        except subprocess.TimeoutExpired:
            os.killpg(srv.pid, signal.SIGKILL)
            srv.wait()
        log.close()
        time.sleep(10)   # the pinned arena and VRAM are given back before the next load
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_config")
    ap.add_argument("exe")
    ap.add_argument("out")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--expert-cache", type=int, default=0)
    ap.add_argument("--src", default=str(Path(__file__).resolve().parents[2]))   # the repository root
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out).resolve()   # the server runs in --src: its config and logs need absolute paths
    out.mkdir(parents=True, exist_ok=True)
    a.exe = str(Path(a.exe).resolve())   # the server resolves a relative exe against the config's cwd, not ours
    base = json.loads(Path(a.base_config).read_text(encoding="utf-8-sig"))
    arms = a.arms.split(",")
    rng = random.Random(a.seed)
    with open(out / "results.jsonl", "a") as res:
        for rep in range(1, a.reps + 1):
            order = arms[:]
            rng.shuffle(order)
            print(f"rep {rep}: {order}", flush=True)
            for arm in order:
                for row in run_load(base, a.exe, arm, rep, out, a):
                    res.write(json.dumps(row) + "\n")
                    res.flush()


if __name__ == "__main__":
    main()
