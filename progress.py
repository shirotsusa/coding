# scripts/progress.py
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

RUNS = Path("lake/bronze/meta/runs/runs.jsonl")
META = Path("lake/bronze/meta")
SPECS = META / "specs"

# 1 job あたりの対象ID数（あなたの分割ルールに合わせる）
DEFAULT_IDS_PER_JOB = {"ROW": 1000, "WAFER": 1000, "LOT": 50}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_runs_rows() -> list[dict]:
    if not RUNS.exists():
        return []
    return [json.loads(l) for l in RUNS.read_text(encoding="utf-8").splitlines() if l.strip()]


def pick_latest_snapshot_from_runs(rows: list[dict]) -> str | None:
    snaps = [r.get("snapshot") for r in rows if r.get("snapshot")]
    return max(snaps) if snaps else None


def best_record(a: dict, b: dict) -> dict:
    # parquet成功 > raw_only > ts新しい
    a_ok = bool(a.get("parquet_path"))
    b_ok = bool(b.get("parquet_path"))
    if a_ok != b_ok:
        return a if a_ok else b
    a_raw = bool(a.get("raw_path"))
    b_raw = bool(b.get("raw_path"))
    if a_raw != b_raw:
        return a if a_raw else b
    return a if a.get("ts", "") >= b.get("ts", "") else b


def fmt_bar(ratio: float, width: int = 40) -> str:
    ratio = 0.0 if ratio < 0 else (1.0 if ratio > 1 else ratio)
    return ("#" * int(ratio * width)).ljust(width, "-")


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


def normalize_edate(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    # 2025/04/17 -> 2025-04-17
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            y, m, d = parts
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    # 20250417 -> 2025-04-17
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def guess_ids_csv_path(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    candidates = [
        Path("ids.csv"),
        META / "ids.csv",
        META / "inputs" / "ids.csv",
        META / "input" / "ids.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def read_ids_counts_by_edate(
    ids_csv: Path,
    *,
    e_date_col: str = "E_DATE",
    lot_col: str = "LOT_ID",
    wafer_col: str = "WAFER_ID",
    wafers_per_lot: int = 25,
    encoding_candidates: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp932", "shift_jis"),
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """
    ids.csv から e_date ごとの counts を作る（列名は大小文字を吸収）。
    戻り値: (lots_by_edate, wafers_by_edate, rows_by_edate)
      - wafer列があれば wafers_by_edate はその行数
      - wafer列が無ければ wafers_by_edate = lots_by_edate * wafers_per_lot
    """
    want_e = e_date_col.strip().upper()
    want_l = lot_col.strip().upper()
    want_w = wafer_col.strip().upper()

    last_err: Exception | None = None
    for enc in encoding_candidates:
        try:
            lots_by = defaultdict(int)
            wafers_by = defaultdict(int)
            rows_by = defaultdict(int)

            with open(ids_csv, "r", encoding=enc, errors="replace", newline="") as f:
                r = csv.DictReader(f)
                if not r.fieldnames:
                    raise ValueError("ids.csv has no header")

                fns_upper = {c.strip().upper() for c in r.fieldnames if c is not None}
                if want_e not in fns_upper:
                    raise ValueError(f"missing column: {want_e} (found={sorted(fns_upper)[:20]}...)")

                has_wafer = want_w in fns_upper
                has_lot = want_l in fns_upper

                for row in r:
                    # 行キーを大文字に正規化
                    row_u = {(k.strip().upper() if k else ""): (v if v is not None else "") for k, v in row.items()}
                    ed = normalize_edate(row_u.get(want_e, ""))
                    if not ed:
                        continue

                    rows_by[ed] += 1

                    if has_wafer and row_u.get(want_w, "").strip():
                        wafers_by[ed] += 1
                    elif has_lot and row_u.get(want_l, "").strip():
                        lots_by[ed] += 1

            if not wafers_by:
                # wafer列が無い運用：lot数×wafers_per_lot で見積もり
                for ed, n_lots in lots_by.items():
                    wafers_by[ed] = n_lots * wafers_per_lot

            return dict(lots_by), dict(wafers_by), dict(rows_by)

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"failed to read ids.csv: {ids_csv}") from last_err


def list_data_types_from_specs() -> list[str]:
    if not SPECS.exists():
        return []
    dts = []
    for p in SPECS.glob("data_type=*"):
        name = p.name
        if name.startswith("data_type="):
            dts.append(name.split("=", 1)[1])
    return sorted(set(dts))


def count_ope_groups(data_type: str) -> int:
    p = SPECS / f"data_type={data_type}" / "enabled_columns.json"
    if not p.exists():
        return 0
    j = read_json(p)
    return len(j.get("groups", []))


def infer_cls_by_data_type_from_runs(rows_all: list[dict], snapshot: str) -> dict[str, str]:
    """
    実行コードを変えない前提なので、clsは runs.jsonl の job_key から推定する。
    未登場 data_type は ROW とみなす（あなたの基本運用に合わせる）。
    """
    cnt: dict[str, Counter] = defaultdict(Counter)
    for r in rows_all:
        if r.get("snapshot") != snapshot:
            continue
        jk = r.get("job_key")
        if not jk:
            continue
        k = parse_job_key(jk)
        dt = k.get("data_type")
        cls = k.get("cls")
        if dt and cls:
            cnt[dt][cls] += 1

    out = {}
    for dt, c in cnt.items():
        out[dt] = c.most_common(1)[0][0]
    return out


def calc_planned_jobs(
    *,
    data_types: list[str],
    lots_by_edate: dict[str, int],
    wafers_by_edate: dict[str, int],
    cls_by_dt: dict[str, str],
    ids_per_job: dict[str, int],
) -> tuple[int, dict[str, int], dict[str, int]]:
    """
    planned_jobs(dt) = ope_groups(dt) * Σ_e_date ceil(n_ids(e_date, cls) / per_job(cls))
    """
    planned_total = 0
    planned_by_dt = {}
    planned_by_cls = defaultdict(int)

    for dt in data_types:
        groups = count_ope_groups(dt)
        if groups == 0:
            continue

        cls = cls_by_dt.get(dt, "ROW")
        per_job = ids_per_job.get(cls)
        if not per_job:
            continue

        sum_chunks = 0
        if cls == "LOT":
            for _, n_lots in lots_by_edate.items():
                if n_lots > 0:
                    sum_chunks += math.ceil(n_lots / per_job)
        else:
            for _, n_waf in wafers_by_edate.items():
                if n_waf > 0:
                    sum_chunks += math.ceil(n_waf / per_job)

        planned = groups * sum_chunks
        planned_by_dt[dt] = planned
        planned_by_cls[cls] += planned
        planned_total += planned

    return planned_total, planned_by_dt, dict(planned_by_cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=str, default=None, help="対象snapshot（省略時はruns.jsonlの最新）")
    ap.add_argument("--ids-csv", type=str, default=None, help="投入予定のids.csvパス（省略時は候補から自動探索）")
    ap.add_argument("--wafers-per-lot", type=int, default=25, help="lot→wafer換算（ids.csvがlot粒度のとき）")

    # ★あなたの ids.csv（LOT_ID / E_DATE）に合わせてデフォルトを大文字に変更
    ap.add_argument("--e-date-col", type=str, default="E_DATE")
    ap.add_argument("--lot-col", type=str, default="LOT_ID")
    ap.add_argument("--wafer-col", type=str, default="WAFER_ID")  # 存在しないなら自動でlot換算に落ちる

    ap.add_argument("--per-job-row", type=int, default=DEFAULT_IDS_PER_JOB["ROW"])
    ap.add_argument("--per-job-wafer", type=int, default=DEFAULT_IDS_PER_JOB["WAFER"])
    ap.add_argument("--per-job-lot", type=int, default=DEFAULT_IDS_PER_JOB["LOT"])
    args = ap.parse_args()

    rows_all = load_runs_rows()
    snap = args.snapshot or pick_latest_snapshot_from_runs(rows_all)
    if not snap:
        print(f"runs.jsonl not found or empty: {RUNS}")
        return

    ids_csv = guess_ids_csv_path(args.ids_csv)
    if not ids_csv:
        print("ids.csv が見つかりません。--ids-csv で投入予定ids.csvを指定してください。")
        return

    lots_by_edate, wafers_by_edate, rows_by_edate = read_ids_counts_by_edate(
        ids_csv,
        e_date_col=args.e_date_col,
        lot_col=args.lot_col,
        wafer_col=args.wafer_col,
        wafers_per_lot=args.wafers_per_lot,
    )
    planned_lots = sum(lots_by_edate.values())
    planned_wafers = sum(wafers_by_edate.values())
    planned_edates = len(set(list(lots_by_edate.keys()) + list(wafers_by_edate.keys())))

    data_types = list_data_types_from_specs()
    if not data_types:
        print(f"meta/specs が見つかりません: {SPECS}")
        return

    cls_by_dt = infer_cls_by_data_type_from_runs(rows_all, snap)

    ids_per_job = {
        "ROW": args.per_job_row,
        "WAFER": args.per_job_wafer,
        "LOT": args.per_job_lot,
    }

    planned_jobs, planned_by_dt, planned_by_cls = calc_planned_jobs(
        data_types=data_types,
        lots_by_edate=lots_by_edate,
        wafers_by_edate=wafers_by_edate,
        cls_by_dt=cls_by_dt,
        ids_per_job=ids_per_job,
    )

    # --- 実績（runs） ---
    by_job: dict[str, dict] = {}
    for r in rows_all:
        if r.get("snapshot") != snap:
            continue
        jk = r.get("job_key")
        if not jk:
            continue
        by_job[jk] = r if jk not in by_job else best_record(by_job[jk], r)

    done = 0
    raw_only = 0
    status_cnt = defaultdict(int)

    done_by_dt = defaultdict(int)
    seen_by_dt = defaultdict(int)

    for jk, r in by_job.items():
        k = parse_job_key(jk)
        dt = k.get("data_type")
        if dt:
            seen_by_dt[dt] += 1

        if r.get("parquet_path"):
            done += 1
            status_cnt["parquet_ok"] += 1
            if dt:
                done_by_dt[dt] += 1
        elif r.get("raw_path"):
            raw_only += 1
            status_cnt["raw_only"] += 1
        else:
            status_cnt[r.get("status", "unknown")] += 1

    seen = len(by_job)

    # --- 出力 ---
    print(f"snapshot: {snap}")
    print(f"ids_csv: {ids_csv}")
    print(f"planned e_dates: {planned_edates}")
    if planned_lots > 0:
        print(f"planned lots:   {planned_lots}")
    print(f"planned wafers: {planned_wafers}")
    print(f"planned data_types (from meta/specs): {len(data_types)}")
    print(f"TOTAL planned jobs: {planned_jobs}")

    if planned_jobs > 0:
        done_ratio = done / planned_jobs
        seen_ratio = seen / planned_jobs
        print("\nProgress (order-independent):")
        print(f"  done (parquet_ok): {done:7d}/{planned_jobs:7d} [{fmt_bar(done_ratio)}] {done_ratio*100:5.1f}%")
        print(f"  seen (attempted):  {seen:7d}/{planned_jobs:7d} [{fmt_bar(seen_ratio)}] {seen_ratio*100:5.1f}%")
        print(f"  raw_only:          {raw_only:7d}")
        if seen > planned_jobs:
            print(f"\n[WARN] seen({seen}) > planned_jobs({planned_jobs}).")
            print("      planned側の推定条件（ids.csv/specs/分割条件）がズレている可能性があります。")
    else:
        print("\nProgress:")
        print("  planned_jobs=0 のため割合を計算できません（meta/specs/enabled_columns.json を確認）")

    if status_cnt:
        print("\nRun status breakdown:")
        for k in sorted(status_cnt.keys()):
            if k not in ("parquet_ok", "raw_only"):
                print(f"  {k}: {status_cnt[k]}")

    if planned_by_cls:
        print("\nPlanned jobs by cls:")
        for cls, v in sorted(planned_by_cls.items(), key=lambda x: -x[1]):
            print(f"  {cls:6s}: {v}")

    if planned_by_dt:
        print("\nBy data_type (planned vs done/seen):")
        for dt, p in sorted(planned_by_dt.items(), key=lambda x: -x[1])[:30]:
            d = done_by_dt.get(dt, 0)
            s = seen_by_dt.get(dt, 0)
            rr = (d / p * 100) if p else 0.0
            print(f"  {dt:12s} planned={p:7d}  done={d:7d}  seen={s:7d}  done%={rr:5.1f}%")


if __name__ == "__main__":
    main()
