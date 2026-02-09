# scripts/progress.py
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime, date


RUNS = Path("lake/bronze/meta/runs/runs.jsonl")
META = Path("lake/bronze/meta")

# あなたの運用に合わせて調整（母数推定に使う）
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


def parse_edate(s: str | None) -> date | None:
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def _get_part_value(path: Path, prefix: str) -> str:
    """
    path（lot_ids.json）から上位を辿って、prefix=... の値を探す。
    階層が変わっても落ちにくくするための保険。
    """
    for p in [path] + list(path.parents):
        name = p.name
        if name.startswith(prefix + "="):
            return name.split("=", 1)[1]
    raise ValueError(f"missing {prefix}=... in path: {path}")


def get_current_edate_from_runs_tail(rows_all: list[dict], snapshot: str) -> tuple[str | None, dict | None]:
    """
    runs.jsonl の末尾から遡って snapshot一致の行を探し、e_date（あれば）を返す。
    e_dateが無い行しか無い場合は job_key から e_date=... を拾う。
    """
    for r in reversed(rows_all):
        if r.get("snapshot") != snapshot:
            continue

        ed = r.get("e_date")
        if ed:
            return str(ed), r

        jk = r.get("job_key")
        if jk and "__e_date=" in jk:
            try:
                ed2 = jk.split("__e_date=", 1)[1].split("__", 1)[0]
                return ed2, r
            except Exception:
                pass

    return None, None


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
    """
    meta/partitions 配下から snapshot に一致する lot_ids.json を列挙する。
    期待するパス例：
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
            # 解析不能はスキップ（issues側で拾う）
            yield None, None, lot_ids_json


def expected_jobs_for_partition(snapshot: str, data_type: str, e_date: str, lot_ids_json: Path) -> tuple[int, dict]:
    """
    期待job数 = (opeグループ数) × ceil(対象ID数 / ids_per_job(cls))
    対象ID数は cls により n_wafers / n_lots を切り替える。
    """
    if not data_type or not e_date:
        return 0, {"reason": "bad_partition_path", "path": str(lot_ids_json)}

    lineage_path = META / "lineage" / f"data_type={data_type}" / f"e_date={e_date}" / f"snapshot={snapshot}" / "lineage.json"
    if not lineage_path.exists():
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
    out = {}
    if len(parts) >= 2:
        out["dataset"] = parts[0]
        out["data_type"] = parts[1]
    for p in parts[2:]:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k] = v
    return out


def load_rows() -> list[dict]:
    if not RUNS.exists():
        return []
    return [json.loads(l) for l in RUNS.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=str, default=None, help="use specific snapshot (default: latest)")
    args = ap.parse_args()

    rows_all = load_rows()
    if not rows_all:
        print("runs.jsonl not found or empty:", RUNS)
        return

    snap = args.snapshot or pick_latest_snapshot(rows_all)
    if not snap:
        print("No snapshot found.")
        return

    # 末尾行から current e_date を拾う（運用上の“進行中目安”）
    current_edate_str, current_rec = get_current_edate_from_runs_tail(rows_all, snap)
    current_edate = parse_edate(current_edate_str) if current_edate_str else None

    # 最新snapshotのrunsを job_key 単位で統合
    rows = [r for r in rows_all if r.get("snapshot") == snap and r.get("job_key")]
    by: dict[str, dict] = {}
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
    exp_detail: list[dict] = []
    issues: list[dict] = []

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

    bar_done = ("#" * int(done_ratio * 40)).ljust(40, "-")
    bar_seen = ("#" * int(seen_ratio * 40)).ljust(40, "-")

    print(f"snapshot: {snap}")
    print(f"expected jobs: {expected}")
    print(f"done (parquet_ok): {done}   remaining: {remaining}")
    print(f"seen (attempted unique jobs): {seen}   raw_only: {raw_only}")
    print(f"[DONE {bar_done}] {done_ratio*100:.1f}%  (done/expected)")
    print(f"[SEEN {bar_seen}] {seen_ratio*100:.1f}%  (seen/expected)")

    for k in sorted(cnt.keys()):
        if k not in ("parquet_ok", "raw_only"):
            print(f"{k}: {cnt[k]}")

    # ---- 追加：日付ベースのざっくり進捗（末尾行e_date vs 最大e_date） ----
    planned_dates = sorted({parse_edate(x["e_date"]) for x in exp_detail if parse_edate(x["e_date"]) is not None})
    max_edate = planned_dates[-1] if planned_dates else None

    if planned_dates:
        print("\nApprox progress by e_date (tail-based):")
        print(f"  current_e_date (tail): {current_edate_str or 'N/A'}")
        print(f"  max_planned_e_date:    {max_edate.isoformat() if max_edate else 'N/A'}")

        if current_edate:
            # 単純：日付の並び上の位置
            idx = -1
            for i, d in enumerate(planned_dates):
                if d <= current_edate:
                    idx = i
            if idx >= 0:
                date_ratio = (idx + 1) / len(planned_dates)
                print(f"  date_position:         {idx+1}/{len(planned_dates)} ({date_ratio*100:.1f}%)")
            else:
                print("  date_position:         0/{} (0.0%)".format(len(planned_dates)))

            # 重み付き：期待jobの累積比（e_date<=current の expected を足す）
            cum_expected = 0
            for x in exp_detail:
                d = parse_edate(x.get("e_date"))
                if d and d <= current_edate:
                    cum_expected += int(x.get("expected", 0))
            weighted_ratio = (cum_expected / expected) if expected else 0.0
            print(f"  weighted_expected:     {cum_expected}/{expected} ({weighted_ratio*100:.1f}%)")

            if current_rec and current_rec.get("job_key"):
                print(f"  current_job_key:       {current_rec['job_key']}")
        else:
            print("  (skip: current_e_date could not be parsed or not found)")
    else:
        print("\nApprox progress by e_date (tail-based):")
        print("  (skip: no planned e_date found from partitions)")

    # data_type別（どこが重いか）
    dt_stat = defaultdict(lambda: {"expected": 0, "done": 0, "seen": 0})
    for info in exp_detail:
        dt_stat[info["data_type"]]["expected"] += int(info["expected"])

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
        def _score(item):
            s = item[1]
            exp = s["expected"]
            return (s["done"] / exp) if exp else -1

        for dt, s in sorted(dt_stat.items(), key=_score):
            exp = s["expected"]
            d = s["done"]
            se = s["seen"]
            rr = (d / exp * 100) if exp else 0.0
            print(f"  {dt:12s}  done/expected={d:7d}/{exp:7d} ({rr:5.1f}%)  seen={se:7d}")

    if issues:
        print("\n[WARN] Some partitions were skipped for expected-count calculation (expected may be undercounted):")
        for x in issues[:12]:
            print(" ", x)
        if len(issues) > 12:
            print(f"  ... and {len(issues)-12} more")


if __name__ == "__main__":
    main()
