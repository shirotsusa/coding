# scripts/progress.py
from __future__ import annotations

import json
import math
from pathlib import Path
from collections import defaultdict

RUNS = Path("lake/bronze/meta/runs/runs.jsonl")
META = Path("lake/bronze/meta")

# あなたの設定に合わせる（必須）
IDS_PER_JOB_BY_CLS = {
    "ROW": 1000,
    "WAFER": 1000,
    "LOT": 50,
}

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def pick_latest_snapshot(rows: list[dict]) -> str | None:
    snaps = [r.get("snapshot") for r in rows if r.get("snapshot")]
    return max(snaps) if snaps else None

def best_record(a: dict, b: dict) -> dict:
    """同一job_keyでの統合: parquet成功 > raw_only > ts新しい"""
    a_ok = bool(a.get("parquet_path"))
    b_ok = bool(b.get("parquet_path"))
    if a_ok != b_ok:
        return a if a_ok else b
    a_raw = bool(a.get("raw_path"))
    b_raw = bool(b.get("raw_path"))
    if a_raw != b_raw:
        return a if a_raw else b
    return a if a.get("ts","") >= b.get("ts","") else b

def resolve_cls(lineage: dict, data_type: str) -> str:
    default_cls = lineage.get("default_cls", "ROW")
    overrides = lineage.get("sheet_to_cls_overrides", {}) or {}
    return overrides.get(data_type, default_cls)

def count_ope_groups(data_type: str) -> int:
    p = META / "specs" / f"data_type={data_type}" / "enabled_columns.json"
    if not p.exists():
        return 0
    j = read_json(p)
    return len(j.get("groups", []))

def iter_partitions(snapshot: str):
    base = META / "partitions"
    if not base.exists():
        return
    # meta/partitions/data_type=<DT>/e_date=<ED>/snapshot=<SNAP>/lot_ids.json
    for lot_ids_json in base.glob(f"data_type=*/e_date=*/snapshot={snapshot}/lot_ids.json"):
        # data_type=<DT>
        data_type = lot_ids_json.parents[3].name.split("=", 1)[1]
        # e_date=<ED>
        e_date = lot_ids_json.parents[2].name.split("=", 1)[1]
        yield data_type, e_date, lot_ids_json

def expected_jobs_for_partition(snapshot: str, data_type: str, e_date: str, lot_ids_json: Path) -> tuple[int, dict]:
    # lineageからclsを決める
    lineage_path = META / "lineage" / f"data_type={data_type}" / f"e_date={e_date}" / f"snapshot={snapshot}" / "lineage.json"
    if not lineage_path.exists():
        # fallback: 期待値計算不能
        return 0, {"reason": "missing_lineage", "data_type": data_type, "e_date": e_date}

    lineage = read_json(lineage_path)
    cls = resolve_cls(lineage, data_type)

    part = read_json(lot_ids_json)
    n_wafers = int(part.get("n_wafers", 0))
    n_lots = int(part.get("n_lots", 0))

    if cls in ("ROW", "WAFER"):
        n_ids = n_wafers
    elif cls == "LOT":
        n_ids = n_lots
    else:
        return 0, {"reason": "unknown_cls", "cls": cls, "data_type": data_type, "e_date": e_date}

    per_job = IDS_PER_JOB_BY_CLS.get(cls)
    if not per_job:
        return 0, {"reason": "missing_ids_per_job", "cls": cls, "data_type": data_type, "e_date": e_date}

    ope_groups = count_ope_groups(data_type)
    if ope_groups == 0:
        return 0, {"reason": "missing_enabled_columns", "data_type": data_type, "e_date": e_date}

    n_chunks = math.ceil(n_ids / per_job) if n_ids > 0 else 0
    exp = ope_groups * n_chunks

    info = {
        "data_type": data_type,
        "e_date": e_date,
        "cls": cls,
        "n_ids": n_ids,
        "per_job": per_job,
        "n_chunks": n_chunks,
        "ope_groups": ope_groups,
        "expected": exp,
    }
    return exp, info

def parse_job_key(jk: str) -> dict:
    # {dataset}__{data_type}__ope=...__cls=...__e_date=...__part=....
    parts = jk.split("__")
    out = {"dataset": parts[0], "data_type": parts[1]}
    for p in parts[2:]:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k] = v
    return out

def main():
    if not RUNS.exists():
        print("runs.jsonl not found:", RUNS)
        return

    rows = [json.loads(l) for l in RUNS.read_text(encoding="utf-8").splitlines() if l.strip()]
    snap = pick_latest_snapshot(rows)
    if not snap:
        print("No snapshot found.")
        return

    # 最新snapshotのrunsを job_key 単位で統合
    rows = [r for r in rows if r.get("snapshot") == snap and r.get("job_key")]
    by = {}
    for r in rows:
        jk = r["job_key"]
        by[jk] = r if jk not in by else best_record(by[jk], r)

    # 完了/失敗内訳
    cnt = defaultdict(int)
    done = 0
    raw_only = 0
    for r in by.values():
        if r.get("parquet_path"):
            done += 1
            cnt["parquet_ok"] += 1
        elif r.get("raw_path"):
            raw_only += 1
            cnt["raw_only"] += 1
        else:
            cnt[r.get("status", "unknown")] += 1

    seen = len(by)

    # 母数（期待job数）を meta/partitions から復元
    expected = 0
    exp_detail = []
    issues = []
    for data_type, e_date, lot_ids_json in iter_partitions(snap):
        exp, info = expected_jobs_for_partition(snap, data_type, e_date, lot_ids_json)
        expected += exp
        if exp > 0:
            exp_detail.append(info)
        else:
            issues.append(info)

    done_ratio = (done / expected) if expected else 0.0
    seen_ratio = (seen / expected) if expected else 0.0
    remaining = max(expected - done, 0)

    # 表示
    bar_done = "#" * int(done_ratio * 40)
    bar_done = bar_done.ljust(40, "-")
    bar_seen = "#" * int(seen_ratio * 40)
    bar_seen = bar_seen.ljust(40, "-")

    print(f"snapshot: {snap}")
    print(f"expected jobs: {expected}")
    print(f"done (parquet_ok): {done}   remaining: {remaining}")
    print(f"seen (attempted unique jobs): {seen}   raw_only: {raw_only}")
    print(f"[DONE {bar_done}] {done_ratio*100:.1f}%  (done/expected)")
    print(f"[SEEN {bar_seen}] {seen_ratio*100:.1f}%  (seen/expected)")
    # 失敗内訳
    for k in sorted(cnt.keys()):
        if k not in ("parquet_ok", "raw_only"):
            print(f"{k}: {cnt[k]}")

    # data_type別（どこが重いかを見る）
    dt_stat = defaultdict(lambda: {"expected": 0, "done": 0, "seen": 0})
    for info in exp_detail:
        dt_stat[info["data_type"]]["expected"] += info["expected"]
    for jk, r in by.items():
        k = parse_job_key(jk)
        dt = k.get("data_type")
        if not dt:
            continue
        dt_stat[dt]["seen"] += 1
        if r.get("parquet_path"):
            dt_stat[dt]["done"] += 1

    if dt_stat:
        print("\nBy data_type:")
        for dt, s in sorted(dt_stat.items(), key=lambda x: (x[1]["done"] / x[1]["expected"]) if x[1]["expected"] else -1):
            exp = s["expected"]
            d = s["done"]
            se = s["seen"]
            rr = (d/exp*100) if exp else 0.0
            print(f"  {dt:10s}  done/expected={d:6d}/{exp:6d} ({rr:5.1f}%)  seen={se:6d}")

    if issues:
        # 母数が計算できないpartitionがあると expected が過小評価になるので警告
        print("\n[WARN] Some partitions were skipped for expected-count calculation:")
        for x in issues[:10]:
            print(" ", x)
        if len(issues) > 10:
            print(f"  ... and {len(issues)-10} more")

if __name__ == "__main__":
    main()
