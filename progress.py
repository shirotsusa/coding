# scripts/progress.py
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime

RUNS = Path("lake/bronze/meta/runs/runs.jsonl")
META = Path("lake/bronze/meta")

# 母数推定に使う「1 job あたりの対象ID数」
# あなたの運用に合わせて調整してください（ROW/WAFERは wafer_id リストを分割する前提）
IDS_PER_JOB_BY_CLS = {
    "ROW": 1000,
    "WAFER": 1000,
    "LOT": 50,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_runs_rows() -> list[dict]:
    if not RUNS.exists():
        return []
    return [json.loads(l) for l in RUNS.read_text(encoding="utf-8").splitlines() if l.strip()]


def pick_latest_snapshot_from_runs(rows: list[dict]) -> str | None:
    snaps = [r.get("snapshot") for r in rows if r.get("snapshot")]
    return max(snaps) if snaps else None


def pick_latest_snapshot_from_partitions() -> str | None:
    base = META / "partitions"
    if not base.exists():
        return None
    snaps = set()
    for p in base.glob("data_type=*/e_date=*/snapshot=*/lot_ids.json"):
        try:
            snaps.add(_get_part_value(p, "snapshot"))
        except Exception:
            pass
    return max(snaps) if snaps else None


def best_record(a: dict, b: dict) -> dict:
    """
    同一job_keyの統合ポリシー：
      parquet成功 > raw_only > ts新しい
    """
    a_ok = bool(a.get("parquet_path"))
    b_ok = bool(b.get("parquet_path"))
    if a_ok != b_ok:
        return a if a_ok else b

    a_raw = bool(a.get("raw_path"))
    b_raw = bool(b.get("raw_path"))
    if a_raw != b_raw:
        return a if a_raw else b

    return a if a.get("ts", "") >= b.get("ts", "") else b


def _get_part_value(path: Path, prefix: str) -> str:
    """
    path（lot_ids.json）から上位を辿って prefix=... の値を探す。
    階層変更に強くするための保険。
    """
    for p in [path] + list(path.parents):
        name = p.name
        if name.startswith(prefix + "="):
            return name.split("=", 1)[1]
    raise ValueError(f"missing {prefix}=... in path: {path}")


def count_ope_groups(data_type: str) -> int:
    p = META / "specs" / f"data_type={data_type}" / "enabled_columns.json"
    if not p.exists():
        return 0
    j = read_json(p)
    return len(j.get("groups", []))


def resolve_cls(snapshot: str, data_type: str, e_date: str) -> str | None:
    lineage_path = META / "lineage" / f"data_type={data_type}" / f"e_date={e_date}" / f"snapshot={snapshot}" / "lineage.json"
    if not lineage_path.exists():
        return None
    lineage = read_json(lineage_path)
    default_cls = lineage.get("default_cls", "ROW")
    overrides = lineage.get("sheet_to_cls_overrides", {}) or {}
    return overrides.get(data_type, default_cls)


def iter_partitions(snapshot: str):
    """
    meta/partitions から snapshot に一致する lot_ids.json を列挙する。
    期待する例：
      meta/partitions/data_type=DS_CHAR/e_date=2026-04-01/snapshot=.../lot_ids.json
    """
    base = META / "partitions"
    if not base.exists():
        return
    for lot_ids_json in base.glob(f"data_type=*/e_date=*/snapshot={snapshot}/lot_ids.json"):
        try:
            data_type = _get_part_value(lot_ids_json, "data_type")
            e_date = _get_part_value(lot_ids_json, "e_date")
            yield data_type, e_date, lot_ids_json
        except Exception:
            yield None, None, lot_ids_json


def expected_jobs_for_partition(snapshot: str, data_type: str, e_date: str, lot_ids_json: Path) -> tuple[int, dict]:
    """
    期待job数 = (opeグループ数) × ceil(対象ID数 / ids_per_job(cls))
    対象ID数：cls=ROW/WAFER -> n_wafers, cls=LOT -> n_lots
    """
    if not data_type or not e_date:
        return 0, {"reason": "bad_partition_path", "path": str(lot_ids_json)}

    part = read_json(lot_ids_json)
    n_wafers = int(part.get("n_wafers", 0))
    n_lots = int(part.get("n_lots", 0))

    cls = resolve_cls(snapshot, data_type, e_date)
    if not cls:
        return 0, {"reason": "missing_lineage", "data_type": data_type, "e_date": e_date}

    per_job = IDS_PER_JOB_BY_CLS.get(cls)
    if not per_job:
        return 0, {"reason": "missing_ids_per_job", "cls": cls, "data_type": data_type, "e_date": e_date}

    ope_groups = count_ope_groups(data_type)
    if ope_groups == 0:
        return 0, {"reason": "missing_enabled_columns", "data_type": data_type, "e_date": e_date}

    if cls in ("ROW", "WAFER"):
        n_ids = n_wafers
    elif cls == "LOT":
        n_ids = n_lots
    else:
        return 0, {"reason": "unknown_cls", "cls": cls, "data_type": data_type, "e_date": e_date}

    n_chunks = math.ceil(n_ids / per_job) if n_ids > 0 else 0
    expected = ope_groups * n_chunks

    info = {
        "data_type": data_type,
        "e_date": e_date,
        "cls": cls,
        "n_wafers": n_wafers,
        "n_lots": n_lots,
        "n_ids": n_ids,
        "per_job": per_job,
        "n_chunks": n_chunks,
        "ope_groups": ope_groups,
        "expected": expected,
    }
    return expected, info


def parse_job_key(jk: str) -> dict:
    # {dataset}__{data_type}__ope=...__cls=...__e_date=...__part=....
    parts = jk.split("__")
    out = {}
    if len(parts) >= 2:
        out["dataset"] = parts[0]
        out["data_type"] = parts[1]
    for p in parts[2:]:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k] = v
    return out


def fmt_bar(ratio: float, width: int = 40) -> str:
    ratio = 0.0 if ratio < 0 else (1.0 if ratio > 1 else ratio)
    return ("#" * int(ratio * width)).ljust(width, "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=str, default=None, help="対象snapshot（省略時はruns.jsonlの最新）")
    args = ap.parse_args()

    rows_all = load_runs_rows()

    # snapshot決定：基本はruns優先、runsが無ければpartitionsから
    snap = args.snapshot
    if not snap:
        snap = pick_latest_snapshot_from_runs(rows_all) if rows_all else None
    if not snap:
        snap = pick_latest_snapshot_from_partitions()

    if not snap:
        print("snapshot が見つかりません（runs.jsonl/partitionsを確認してください）")
        return

    # --- 実績（runs）集計 ---
    by_job: dict[str, dict] = {}
    if rows_all:
        rows = [r for r in rows_all if r.get("snapshot") == snap and r.get("job_key")]
        for r in rows:
            jk = r["job_key"]
            by_job[jk] = r if jk not in by_job else best_record(by_job[jk], r)

    done = 0
    raw_only = 0
    status_cnt = defaultdict(int)

    for r in by_job.values():
        if r.get("parquet_path"):
            done += 1
            status_cnt["parquet_ok"] += 1
        elif r.get("raw_path"):
            raw_only += 1
            status_cnt["raw_only"] += 1
        else:
            status_cnt[r.get("status", "unknown")] += 1

    seen = len(by_job)

    # --- 母数（planned/expected）集計 ---
    expected_total = 0
    planned_partitions = 0
    total_wafers = 0
    total_lots = 0

    exp_detail: list[dict] = []
    issues: list[dict] = []
    expected_by_cls = defaultdict(int)
    expected_by_data_type = defaultdict(int)

    for data_type, e_date, lot_ids_json in iter_partitions(snap):
        exp, info = expected_jobs_for_partition(snap, data_type, e_date, lot_ids_json)
        if exp > 0:
            expected_total += exp
            planned_partitions += 1
            total_wafers += int(info.get("n_wafers", 0))
            total_lots += int(info.get("n_lots", 0))
            exp_detail.append(info)
            expected_by_cls[str(info["cls"])] += exp
            expected_by_data_type[str(info["data_type"])] += exp
        else:
            issues.append(info)

    # --- 出力 ---
    print(f"snapshot: {snap}")
    print(f"planned partitions (data_type×e_date): {planned_partitions}")
    print(f"planned lots:   {total_lots}")
    print(f"planned wafers: {total_wafers}")
    print(f"TOTAL expected jobs: {expected_total}")

    if expected_total > 0:
        done_ratio = done / expected_total
        seen_ratio = seen / expected_total
        print(f"\nProgress:")
        print(f"  done (parquet_ok): {done:7d}/{expected_total:7d} [{fmt_bar(done_ratio)}] {done_ratio*100:5.1f}%")
        print(f"  seen (attempted):  {seen:7d}/{expected_total:7d} [{fmt_bar(seen_ratio)}] {seen_ratio*100:5.1f}%")
        print(f"  raw_only:          {raw_only:7d}")
    else:
        print("\nProgress:")
        print("  expected_total=0 のため割合を計算できません（meta/specs/lineage/partitionsの欠落を疑ってください）")
        print(f"  done(parquet_ok): {done}, seen: {seen}, raw_only: {raw_only}")

    # 失敗内訳
    if status_cnt:
        print("\nRun status breakdown (runs.jsonl):")
        for k in sorted(status_cnt.keys()):
            if k not in ("parquet_ok", "raw_only"):
                print(f"  {k}: {status_cnt[k]}")

    # cls別の予定job数
    if expected_by_cls:
        print("\nExpected jobs by cls:")
        for cls, v in sorted(expected_by_cls.items(), key=lambda x: -x[1]):
            print(f"  {cls:6s}: {v}")

    # data_type別：予定と実績を並べる
    # 実績側（done/seen）をdata_type別に数える
    done_by_dt = defaultdict(int)
    seen_by_dt = defaultdict(int)
    for jk, r in by_job.items():
        k = parse_job_key(jk)
        dt = k.get("data_type")
        if not dt:
            continue
        seen_by_dt[dt] += 1
        if r.get("parquet_path"):
            done_by_dt[dt] += 1

    if expected_by_data_type:
        print("\nBy data_type (expected vs done/seen):")
        # expectedが多い順に表示
        for dt, exp in sorted(expected_by_data_type.items(), key=lambda x: -x[1])[:30]:
            d = done_by_dt.get(dt, 0)
            s = seen_by_dt.get(dt, 0)
            rr = (d / exp * 100) if exp else 0.0
            print(f"  {dt:12s} expected={exp:8d}  done={d:8d}  seen={s:8d}  done%={rr:5.1f}%")

    # 計算から漏れたpartitionがあると expected_total が過小評価になるので警告
    if issues:
        print("\n[WARN] Some partitions were skipped (expected_total may be undercounted):")
        for x in issues[:12]:
            print(" ", x)
        if len(issues) > 12:
            print(f"  ... and {len(issues)-12} more")


if __name__ == "__main__":
    main()
