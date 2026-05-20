#!/usr/bin/env python3
"""Parse [D1] INFO lines from D1 test_pp runs (per-node log files) and produce summary.

Usage:
  python d1_analyze.py /tmp/d1_test_pp_rank0.log /tmp/d1_test_pp_rank1.log
"""
import re, sys, collections

PAT = re.compile(r"\[D1\] (?P<event>\S+)(?:.*?)(?P<rest>$)")

def parse(path):
    counts = collections.Counter()
    first_seen = {}
    last_seen = {}
    samples = collections.defaultdict(list)
    progress_calls_max = -1
    with open(path) as f:
        for ln in f:
            m = re.search(r"\[D1\]\s+(\w+(?:->plugin)?\s*\w*)", ln)
            if not m:
                continue
            ev = m.group(1).strip()
            counts[ev] += 1
            if ev not in first_seen:
                first_seen[ev] = ln.strip()
                samples[ev].append(ln.strip())
            last_seen[ev] = ln.strip()
            if "ncclGinProxyProgress" in ev:
                cm = re.search(r"call#(\d+)", ln)
                if cm:
                    progress_calls_max = max(progress_calls_max, int(cm.group(1)))
    return counts, first_seen, last_seen, progress_calls_max

def main():
    for p in sys.argv[1:]:
        print(f"\n===== {p} =====")
        try:
            counts, first, last, progress_max = parse(p)
        except FileNotFoundError:
            print("  (log missing)")
            continue
        if not counts:
            print("  no [D1] lines — NCCL didn't load patched build OR proxy thread never ran")
            continue
        print(f"  event counts:")
        for ev, n in counts.most_common():
            print(f"    {ev:50s} {n:10d}")
        print(f"  proxy progress max call#: {progress_max}")
        for ev in counts:
            print(f"  first {ev!r}: {first[ev]}")
            if first[ev] != last[ev]:
                print(f"  last  {ev!r}: {last[ev]}")

if __name__ == "__main__":
    main()
