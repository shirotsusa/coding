# scripts/progress.py
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

# ---- Path base: script location (watchでも崩れない) ----
ROOT = Path(__file__).resolve().parents[1]  # <repo>/scripts/progress.py を想定
RUNS = ROOT / "lake/bronze/meta/runs/runs.jsonl"
META = ROOT / "lake/bronze/meta"
SPECS = META / "specs"

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
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            y, m, d = parts
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def guess_ids_csv_path(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        return p if p.exists() else None

    candidates = [
        ROOT / "ids.csv",
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
    debug: bool = False,
) -> tuple[dict[str, int], dict[str, int], str]:
    """
    ids.csv から e_date ごとの lot数/wafer数を作る（列名は大小文字吸収）。
    戻り値: (lots_by_edate, wafers_by_edate, used_encoding)
      - WAFER_ID 列があれば wafers_by_edate はその行数
      - 無ければ wafers_by_edate = lots_by_edate * wafers_per_lot
    """
    want_e = e_date_col.strip().upper()
    want_l = lot_col.strip().upper()
    want_w = wafer_col.strip().upper()

    last_err: Exception | None = None
    for enc in encoding_candidates:
        try:
            lots_by = defaultdict(int)
            wafers_by = defaultdict(int)

            total_rows = 0

            with open(ids_csv, "r", encoding=enc, errors="replace", newline="") as f:
                r = csv.DictReader(f)
                if not r.fieldnames:
                    raise ValueError("ids.csv has no header")

                fns_upper = {c.strip().upper() for c in r.fieldnames if c is not None}
                if want_e not in fns_upper:
                    raise ValueError(f"missing column: {want_e} (found={sorted(fns_upper)[:30]})")

                has_wafer = want_w in fns_upper
                has_lot = want_l in fns_upper

                for row in r:
                    total_rows += 1
                    row_u = {(k.strip().upper() if k else ""): (v if v is not None else "") for k, v in row.items()}
                    ed = normalize_edate(row_u.get(want_e, ""))
                    if not ed:
                        continue

                    if has_wafer and row_u.get(want_w, "").strip():
                        wafers_by[ed] += 1
                    elif has_lot and row_u.get(want_l, "").strip():
                        lots_by[ed] += 1

            if not wafers_by:
                # wafer列が無い運用：lot数×wafers_per_lot で見積もり
                for ed, n_lots in lots_by.items():
                    wafers_by[ed] = n_lots * wafers_per_lot

            if debug:
                print(f"[DEBUG] ids.csv encoding={enc}, total_rows={total_rows}, e_dates={len(set(list(lots_by)+list(wafers_by)))}")

            return dict(lots_by), dict(wafers_by), enc

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
    """
    enabled_columns.json の構造揺れに強くopeグループ数を数える。
    想定例:
      - {"groups":[{"ope":"...", "cols":[...]}, ...]}
      - {"enabled_columns": {"OPE1":[...], "OPE2":[...]}}
      - {"OPE1":[...], "OPE2":[...]}  (トップがdict)
    """
    p = SPECS / f"data_type={data_type}" / "enabled_columns.json"
    if not p.exists():
        return 0

    try:
        j = read_json(p)

        if isinstance(j, dict):
            if "groups" in j and isinstance(j["groups"], list):
                return len(j["groups"])
            if "ope_groups" in j and isinstance(j["ope_groups"], list):
                return len(j["ope_groups"])
            if "opes" in j and isinstance(j["opes"], list):
                return len(j["opes"])
            if "enabled_columns" in j:
                ec = j["enabled_columns"]
                if isinstance(ec, dict):
                    return len(ec.keys())
                if isinstance(ec, list):
                    return len(ec)
            # top-level dict as ope->cols
            # ただしメタキーが混ざるかもなので、値がlist/dictっぽいキーだけ数える
            keys = []
            for k, v in j.items():
                if k in ("meta", "version", "created_at"):
                    continue
                if isinstance(v, (list, dict)):
                    keys.append(k)
            if keys:
                return len(keys)

        if isinstance(j, list):
            # [{"ope":..}, ...] など
            return len(j)

    except Exception:
        return 0

    return 0


def estimate_ope_groups_from_runs(rows_all: list[dict], snapshot: str) -> dict[str, int]:
    """
    specsが取れない時の保険：runsの job_key から dt別のopeユニーク数を推定。
    （実行途中だと過小推定になり得る点には注意）
    """
    seen = defaultdict(set)
    for r in rows_all:
        if r.get("snapshot") != snapshot:
            continue
        jk = r.get("job_key")
        if not jk:
            continue
        k = parse_job_key(jk)
        dt = k.get("data_type")
        ope = k.get("ope")
        if dt and ope:
            seen[dt].add(ope)
    return {dt: len(s) for dt, s in seen.items()}


def infer_cls_by_data_type_from_runs(rows_all: list[dict], snapshot: str) -> dict[str, str]:
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
    snapshot: str,
    rows_all: list[dict],
    data_types: list[str],
    lots_by_edate: dict[str, int],
    wafers_by_edate: dict[str, int],
    cls_by_dt: dict[str, str],
    ids_per_job: dict[str, int],
    fallback_groups: int = 1,
    debug: bool = False,
) -> tuple[int, dict[str, int], dict[str, int], list[str]]:
    """
    planned_jobs(dt) = groups(dt) * Σ_e_date ceil(n_ids(e_date, cls) / per_job(cls))
    groupsは specs優先。0なら runsからopeユニーク数推定。最後にfallback_groups。
    """
    planned_total = 0
    planned_by_dt = {}
    planned_by_cls = defaultdict(int)
    warn = []

    groups_from_runs = estimate_ope_groups_from_runs(rows_all, snapshot)

    for dt in data_types:
        g = count_ope_groups(dt)
        if g <= 0:
            g = groups_from_runs.get(dt, 0)
            if g > 0:
                warn.append(f"[WARN] groups for {dt} inferred from runs (unique ope={g})")
        if g <= 0 and fallback_groups > 0:
            g = fallback_groups
            warn.append(f"[WARN] groups for {dt} fallback to {fallback_groups} (specs/runs unavailable)")

        cls = cls_by_dt.get(dt, "ROW")
        per_job = ids_per_job.get(cls)
        if not per_job:
            warn.append(f"[WARN] per_job missing for cls={cls} (dt={dt}) -> skip")
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

        planned = g * sum_chunks
        planned_by_dt[dt] = planned
        planned_by_cls[cls] += planned
        planned_total += planned

        if debug:
            print(f"[DEBUG] dt={dt} cls={cls} groups={g} sum_chunks={sum_chunks} planned={planned}")

    return planned_total, planned_by_dt, dict(planned_by_cls), warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=str, default=None, help="対象snapshot（省略時はruns.jsonlの最新）")
    ap.add_argument("--ids-csv", type=str, default=None, help="投入予定ids.csv（省略時は候補から探索）")
    ap.add_argument("--wafers-per-lot", type=int, default=25, help="lot→wafer換算（WAFER_ID列が無い場合）")

    # ids.csv が LOT_ID / E_DATE なのでデフォルトを大文字に
    ap.add_argument("--e-date-col", type=str, default="E_DATE")
    ap.add_argument("--lot-col", type=str, default="LOT_ID")
    ap.add_argument("--wafer-col", type=str, default="WAFER_ID")

    ap.add_argument("--per-job-row", type=int, default=DEFAULT_IDS_PER_JOB["ROW"])
    ap.add_argument("--per-job-wafer", type=int, default=DEFAULT_IDS_PER_JOB["WAFER"])
    ap.add_argument("--per-job-lot", type=int, default=DEFAULT_IDS_PER_JOB["LOT"])

    ap.add_argument("--fallback-groups", type=int, default=1, help="specs/runsからgroups取れない場合の最終fallback")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    rows_all = load_runs_rows()
    snap = args.snapshot or pick_latest_snapshot_from_runs(rows_all)
    if not snap:
        print(f"runs.jsonl not found or empty: {RUNS}")
        return

    ids_csv = guess_ids_csv_path(args.ids_csv)
    if not ids_csv:
        print("ids.csv が見つかりません。--ids-csv で指定してください。")
        return

    lots_by_edate, wafers_by_edate, used_enc = read_ids_counts_by_edate(
        ids_csv,
        e_date_col=args.e_date_col,
        lot_col=args.lot_col,
        wafer_col=args.wafer_col,
        wafers_per_lot=args.wafers_per_lot,
        debug=args.debug,
    )
    planned_lots = sum(lots_by_edate.values())
    planned_wafers = sum(wafers_by_edate.values())
    planned_edates = len(set(list(lots_by_edate.keys()) + list(wafers_by_edate.keys())))

    # data_type は specs を基本にする（固定入力側）
    data_types = list_data_types_from_specs()
    if not data_types:
        # 最悪 runs から推定
        dts = set()
        for r in rows_all:
            if r.get("snapshot") != snap:
                continue
            jk = r.get("job_key")
            if not jk:
                continue
            dts.add(parse_job_key(jk).get("data_type"))
        data_types = sorted([d for d in dts if d])
        print(f"[WARN] meta/specsが見えないため data_types を runs から推定: {len(data_types)}")

    cls_by_dt = infer_cls_by_data_type_from_runs(rows_all, snap)

    ids_per_job = {"ROW": args.per_job_row, "WAFER": args.per_job_wafer, "LOT": args.per_job_lot}

    planned_jobs, planned_by_dt, planned_by_cls, warn = calc_planned_jobs(
        snapshot=snap,
        rows_all=rows_all,
        data_types=data_types,
        lots_by_edate=lots_by_edate,
        wafers_by_edate=wafers_by_edate,
        cls_by_dt=cls_by_dt,
        ids_per_job=ids_per_job,
        fallback_groups=args.fallback_groups,
        debug=args.debug,
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
    print(f"runs: {RUNS}")
    print(f"ids_csv: {ids_csv} (encoding={used_enc})")
    print(f"planned e_dates: {planned_edates}")
    print(f"planned lots:   {planned_lots}")
    print(f"planned wafers: {planned_wafers}")
    print(f"planned data_types: {len(data_types)} (from specs if available)")
    print(f"TOTAL planned jobs: {planned_jobs}")

    for w in warn[:10]:
        print(w)
    if len(warn) > 10:
        print(f"[WARN] ... and {len(warn)-10} more warnings")

    if planned_jobs > 0:
        done_ratio = done / planned_jobs
        seen_ratio = seen / planned_jobs
        print("\nProgress (order-independent):")
        print(f"  done (parquet_ok): {done:7d}/{planned_jobs:7d} [{fmt_bar(done_ratio)}] {done_ratio*100:5.1f}%")
        print(f"  seen (attempted):  {seen:7d}/{planned_jobs:7d} [{fmt_bar(seen_ratio)}] {seen_ratio*100:5.1f}%")
        print(f"  raw_only:          {raw_only:7d}")
        if seen > planned_jobs:
            print(f"\n[WARN] seen({seen}) > planned_jobs({planned_jobs}).")
            print("      planned側の推定条件（opeグループ数/分割条件）がズレている可能性があります。")
    else:
        print("\n[ERROR] planned_jobs=0 です。以下が原因の可能性大です：")
        print("  - planned lots/wafers が 0（ids.csv の列名/内容/パス）")
        print("  - enabled_columns.json の読取で opeグループ数が取れていない（specs）")
        print("  - specsのdata_typeが空（meta/specsの配置）")
        print("\n診断ヒント: --debug を付けて dtごとの groups/chunks を表示してください。")

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
