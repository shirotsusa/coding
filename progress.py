# scripts/progress.py
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

RUNS = Path("lake/bronze/meta/runs/runs.jsonl")

def pick_latest_snapshot(rows):
    snaps = [r.get("snapshot") for r in rows if r.get("snapshot")]
    return max(snaps) if snaps else None

def best_record(a, b):
    """同一job_keyのレコード統合: parquet成功 > raw_only > その他は新しいts"""
    a_ok = bool(a.get("parquet_path"))
    b_ok = bool(b.get("parquet_path"))
    if a_ok != b_ok:
        return a if a_ok else b
    a_raw = bool(a.get("raw_path"))
    b_raw = bool(b.get("raw_path"))
    if a_raw != b_raw:
        return a if a_raw else b
    # tsで新しい方
    ta = a.get("ts", "")
    tb = b.get("ts", "")
    return a if ta >= tb else b

def main():
    rows = []
    for line in RUNS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    snap = pick_latest_snapshot(rows)
    if not snap:
        print("No snapshot found.")
        return

    rows = [r for r in rows if r.get("snapshot") == snap and r.get("job_key")]
    by = {}
    for r in rows:
        jk = r["job_key"]
        by[jk] = r if jk not in by else best_record(by[jk], r)

    cnt = defaultdict(int)
    for r in by.values():
        if r.get("parquet_path"):
            cnt["parquet_ok"] += 1
        elif r.get("raw_path"):
            cnt["raw_only"] += 1
        else:
            cnt[r.get("status", "unknown")] += 1

    total_seen = len(by)
    done = cnt["parquet_ok"]
    ratio = (done / total_seen) if total_seen else 0.0
    bar = "#" * int(ratio * 40)
    bar = bar.ljust(40, "-")

    print(f"snapshot: {snap}")
    print(f"jobs seen: {total_seen}  done(parquet): {done}  raw_only: {cnt['raw_only']}")
    print(f"[{bar}] {ratio*100:.1f}%  (done/seen)")
    # 代表的失敗の内訳
    for k in sorted(cnt.keys()):
        if k not in ("parquet_ok", "raw_only"):
            print(f"{k}: {cnt[k]}")

if __name__ == "__main__":
    main()
# scripts/progress.py
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

RUNS = Path("lake/bronze/meta/runs/runs.jsonl")

def pick_latest_snapshot(rows):
    snaps = [r.get("snapshot") for r in rows if r.get("snapshot")]
    return max(snaps) if snaps else None

def best_record(a, b):
    """同一job_keyのレコード統合: parquet成功 > raw_only > その他は新しいts"""
    a_ok = bool(a.get("parquet_path"))
    b_ok = bool(b.get("parquet_path"))
    if a_ok != b_ok:
        return a if a_ok else b
    a_raw = bool(a.get("raw_path"))
    b_raw = bool(b.get("raw_path"))
    if a_raw != b_raw:
        return a if a_raw else b
    # tsで新しい方
    ta = a.get("ts", "")
    tb = b.get("ts", "")
    return a if ta >= tb else b

def main():
    rows = []
    for line in RUNS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    snap = pick_latest_snapshot(rows)
    if not snap:
        print("No snapshot found.")
        return

    rows = [r for r in rows if r.get("snapshot") == snap and r.get("job_key")]
    by = {}
    for r in rows:
        jk = r["job_key"]
        by[jk] = r if jk not in by else best_record(by[jk], r)

    cnt = defaultdict(int)
    for r in by.values():
        if r.get("parquet_path"):
            cnt["parquet_ok"] += 1
        elif r.get("raw_path"):
            cnt["raw_only"] += 1
        else:
            cnt[r.get("status", "unknown")] += 1

    total_seen = len(by)
    done = cnt["parquet_ok"]
    ratio = (done / total_seen) if total_seen else 0.0
    bar = "#" * int(ratio * 40)
    bar = bar.ljust(40, "-")

    print(f"snapshot: {snap}")
    print(f"jobs seen: {total_seen}  done(parquet): {done}  raw_only: {cnt['raw_only']}")
    print(f"[{bar}] {ratio*100:.1f}%  (done/seen)")
    # 代表的失敗の内訳
    for k in sorted(cnt.keys()):
        if k not in ("parquet_ok", "raw_only"):
            print(f"{k}: {cnt[k]}")

if __name__ == "__main__":
    main()
