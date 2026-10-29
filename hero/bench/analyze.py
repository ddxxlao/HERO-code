"""Summarize an ab_driver.py run: per-prompt median decode tok/s per arm, each arm against `fixed` (geometric mean of
the per-prompt ratios, and in how many prompts it was faster), whether the greedy outputs match, and the last fitted
model line of each load.

usage: python analyze.py RUN_DIR
"""
import collections
import json
import math
import re
import statistics
import sys
from pathlib import Path

ORDER = ["cpu", "gpu", "fixed", "balance", "hero"]


def main():
    run = Path(sys.argv[1])
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines() if line.strip()]
    tok = collections.defaultdict(list)
    shas = collections.defaultdict(set)
    for r in rows:
        if r.get("decode_tok_s"):
            tok[(r["arm"], r["prompt"])].append(r["decode_tok_s"])
        shas[r["prompt"]].add((r["arm"], r["sha"]))
    arms = [a for a in ORDER if any(k[0] == a for k in tok)]
    prompts = sorted({k[1] for k in tok})
    med = {k: statistics.median(v) for k, v in tok.items()}

    print("median decode tok/s (n reps)")
    print(f"{'prompt':<9}" + "".join(f"{a:>13}" for a in arms))
    for p in prompts:
        print(f"{p:<9}" + "".join(f"{med.get((a, p), float('nan')):>9.1f} ({len(tok.get((a, p), []))})" for a in arms))

    if "fixed" in arms:
        print("\nagainst fixed (HERO default share): geometric mean of per-prompt ratios, prompts faster")
        for a in arms:
            ratios = [med[(a, p)] / med[("fixed", p)] for p in prompts if (a, p) in med and ("fixed", p) in med]
            if ratios:
                gm = math.exp(sum(map(math.log, ratios)) / len(ratios))
                print(f"  {a:<8} {gm:6.3f}x  ({sum(r > 1 for r in ratios)}/{len(ratios)} faster)")
        # run-to-run spread of `fixed` itself, as the noise floor for the ratios above
        spreads = [(max(v) - min(v)) / statistics.median(v) for (a, p), v in tok.items() if a == "fixed" and len(v) > 1]
        if spreads:
            print(f"  noise floor: fixed's own max-min spread across reps, median {100 * statistics.median(spreads):.1f}%")

    print("\ngreedy outputs: distinct texts per prompt across all loads")
    for p in prompts:
        by_arm = collections.defaultdict(set)
        for a, s in shas[p]:
            by_arm[a].add(s)
        print(f"  {p:<9} {len({s for _, s in shas[p]})} distinct; per arm " +
              ", ".join(f"{a}:{len(by_arm[a])}" for a in arms))

    print("\nfitted models (last line of each load)")
    pat = re.compile(r"(hero model|pcie model|PCIe share): .*")
    for tag in sorted({log.name.split(".")[0] for log in run.glob("*.log")}):
        hits = [m.group(0) for log in sorted(run.glob(f"{tag}.*.log"))
                for line in log.read_text(errors="replace").splitlines() for m in [pat.search(line)] if m]
        if hits:
            print(f"  {tag:<12} {hits[-1][:170]}")


if __name__ == "__main__":
    main()
