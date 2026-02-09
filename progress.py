# scripts/progress.py
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


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

    # "2025-04-17 06:13:38" / "2025/04/17 06:13:38" などは日付部分だけ
    s = s.split()[0].strip()

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


def normalize_colname(s: str) -> str:
    # UTF-8 BOM(\ufeff)がヘッダに付くケースの吸収
    return (s or "").strip().lstrip("\ufeff").upper()


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


def sniff_dialect(sample_text: str, fallback_delim: str = ",") -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample_text, delimiters=[",", "\t", ";", "|"])
    except Exception:
        class D(csv.Dialect):
            delimiter = fallback_delim
            quotechar = '"'
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
        return D()


def read_ids_counts_by_edate(
    ids_csv: Path,
    *,
    e_date_col: str = "E_DATE",
    lot_col: str = "LOT_ID",
    wafer_col: str = "WAFER_ID",
    wafers_per_lot: int = 25,
    encoding_candidates: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp932", "shift_jis"),
    debug: bool = False,
) -> tuple[dict[str, int], dict[str, int], dict]:
    """
    ids.csv から e_date ごとの lot数/wafer数を作る（列名は大小文字＆BOM吸収）。
    戻り値: (lots_by_edate, wafers_by_edate, info)

    - WAFER_ID列があれば「ユニーク wafer_id 数」を採用
    - 無ければ「ユニーク lot_id 数 × wafers_per_lot」で wafer数を見積もり
    """
    want_e = normalize_colname(e_date_col)
    want_l = normalize_colname(lot_col)
    want_w = normalize_colname(wafer_col)

    last_err: Exception | None = None

    for enc in encoding_candidates:
        try:
            # dialect 推定用のサンプル読み
            with open(ids_csv, "r", encoding=enc, errors="replace", newline="") as f:
                sample_lines = []
                for _ in range(60):
                    ln = f.readline()
                    if not ln:
                        break
                    if ln.strip():
                        sample_lines.append(ln)
                sample = "".join(sample_lines)

            dialect = sniff_dialect(sample)

            lots_set_by = defaultdict(set)    # e_date -> set(lot_id)
            wafers_set_by = defaultdict(set)  # e_date -> set(wafer_id)
            total_rows = 0
            rows_with_edate = 0
            rows_with_lot = 0
            rows_with_both = 0

            with open(ids_csv, "r", encoding=enc, errors="replace", newline="") as f:
                r = csv.DictReader(f, dialect=dialect)
                if not r.fieldnames:
                    raise ValueError("ids.csv has no header")

                header_norm = [normalize_colname(h) for h in r.fieldnames if h is not None]
                fns = set(header_norm)

                if want_e not in fns:
                    raise ValueError(f"missing column: {want_e} (found={header_norm[:30]})")

                has_wafer = want_w in fns
                has_lot = want_l in fns

                for row in r:
                    total_rows += 1
                    row_u = {normalize_colname(k): (v if v is not None else "") for k, v in row.items()}

                    ed_raw = str(row_u.get(want_e, "")).strip()
                    ed = normalize_edate(ed_raw)
                    ed_ok = bool(ed)

                    lot_raw = str(row_u.get(want_l, "")).strip()
                    lot_ok = bool(lot_raw)

                    waf_raw = str(row_u.get(want_w, "")).strip()
                    waf_ok = bool(waf_raw)

                    if ed_ok:
                        rows_with_edate += 1
                    if lot_ok:
                        rows_with_lot += 1
                    if ed_ok and lot_ok:
                        rows_with_both += 1

                    if not ed_ok:
                        continue

                    if has_wafer and waf_ok:
                        wafers_set_by[ed].add(waf_raw)
                    elif has_lot and lot_ok:
                        lots_set_by[ed].add(lot_raw)

            lots_by_edate = {ed: len(s) for ed, s in lots_set_by.items()}

            if has_wafer and wafers_set_by:
                wafers_by_edate = {ed: len(s) for ed, s in wafers_set_by.items()}
            else:
                # lot粒度のみ：lot数×25（デフォルト）
                wafers_by_edate = {ed: n_lots * wafers_per_lot for ed, n_lots in lots_by_edate.items()}

            info = {
                "encoding": enc,
                "delimiter": getattr(dialect, "delimiter", ","),
                "header_norm": header_norm,
                "total_rows": total_rows,
                "rows_with_edate": rows_with_edate,
                "rows_with_lot": rows_with_lot,
                "rows_with_both": rows_with_both,
                "has_wafer_col": has_wafer,
                "has_lot_col": has_lot,
            }

            if debug:
                # 最低限の診断だけ（--debug時のみ）
                print(f"[DEBUG] ids_csv={ids_csv}")
                print(f"[DEBUG] ids encoding={enc} delimiter={repr(info['delimiter'])}")
                print(f"[DEBUG] ids header_norm={header_norm[:20]}")
                print(f"[DEBUG] ids rows total={total_rows}, with_edate={rows_with_edate}, with_lot={rows_with_lot}, with_both={rows_with_both}")
                print(f"[DEBUG] ids unique e_dates={len(set(list(lots_by_edate.keys()) + list(wafers_by_edate.keys())))}")
                print(f"[DEBUG] ids planned lots(sum unique per e_date)={sum(lots_by_edate.values())}")
                print(f"[DEBUG] ids planned wafers(est)={sum(wafers_by_edate.values())}")

            return lots_by_edate, wafers_by_edate, info

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
    """
    p = SPECS / f"data_type={data_type}" / "enabled_columns.json"
    if not p.exists():
        return 0

    try:
        j = read_json(p)

        if isinstance(j, dict):
            if "groups" in j and isinstance(j["groups"], list):
                return len(j["groups"])

            if "enabled_columns" in j:
                ec = j["enabled_columns"]
                if isinstance(ec, dict):
                    return len(ec.keys())
                if isinstance(ec, list):
                    return len(ec)

            # top-level dict as ope->cols（メタキー除外、値がlist/dictのキーのみ数える）
            keys = []
            for k, v in j.items():
                if k in ("meta", "version", "created_at"):
                    continue
                if isinstance(v, (list, dict)):
                    keys.append(k)
            if keys:
                return len(keys)

        if isinstance(j, list):
            return len(j)

    except Exception:
        return 0

    return 0


def estimate_ope_groups_from_runs(rows_all: list[dict], snapshot: str) -> dict[str, int]:
    """
    specsから取れない場合の保険：runsの job_key から dt別のopeユニーク数を推定
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
    """
    実行コードを変えない前提：cls は runs.jsonl の job_key から推定する。
    未登場 data_type は ROW 扱い。
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
    snapshot: str,
    rows_all: list[dict],
    data_types: list[str],
    lots_by_edate: dict[str, int],
    wafers_by_edate: dict[str, int],
    cls_by_dt: dict[str, str],
    ids_per_job: dict[str, int],
    fallback_groups: int = 1,
    debug: bool = False,
) -> tuple[int, dict[str, int], dict[str, int]]:
    """
    planned_jobs(dt) = groups(dt) * Σ_e_date ceil(n_ids(e_date, cls) / per_job(cls))
    """
    planned_total = 0
    planned_by_dt = {}
    planned_by_cls = defaultdict(int)

    groups_from_runs = estimate_ope_groups_from_runs(rows_all, snapshot)

    for dt in data_types:
        g = count_ope_groups(dt)
        if g <= 0:
            g = groups_from_runs.get(dt, 0)
        if g <= 0:
            g = fallback_groups

        cls = cls_by_dt.get(dt, "ROW")
        per_job = ids_per_job.get(cls)
        if not per_job:
            if debug:
                print(f"[DEBUG] skip dt={dt} (unknown per_job for cls={cls})")
            continue

        # e_date毎にceil（重要）
        sum_chunks = 0
        if cls == "LOT":
            for n_lots in lots_by_edate.values():
                if n_lots > 0:
                    sum_chunks += math.ceil(n_lots / per_job)
        else:
            for n_waf in wafers_by_edate.values():
                if n_waf > 0:
                    sum_chunks += math.ceil(n_waf / per_job)

        planned = g * sum_chunks
        planned_by_dt[dt] = planned
        planned_by_cls[cls] += planned
        planned_total += planned

        if debug:
            src = "specs" if count_ope_groups(dt) > 0 else ("runs" if groups_from_runs.get(dt, 0) > 0 else "fallback")
            print(f"[DEBUG] dt={dt} cls={cls} groups={g}({src}) sum_chunks={sum_chunks} planned={planned}")

    return planned_total, planned_by_dt, dict(planned_by_cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=str, default=None, help="対象snapshot（省略時はruns.jsonlの最新）")
    ap.add_argument("--ids-csv", type=str, default=None, help="投入予定ids.csv（省略時は候補から探索）")
    ap.add_argument("--wafers-per-lot", type=int, default=25)

    # ids.csv: LOT_ID / E_DATE（BOMありでもOK）
    ap.add_argument("--e-date-col", type=str, default="E_DATE")
    ap.add_argument("--lot-col", type=str, default="LOT_ID")
    ap.add_argument("--wafer-col", type=str, default="WAFER_ID")  # 無ければlot換算に落ちる

    ap.add_argument("--per-job-row", type=int, default=DEFAULT_IDS_PER_JOB["ROW"])
    ap.add_argument("--per-job-wafer", type=int, default=DEFAULT_IDS_PER_JOB["WAFER"])
    ap.add_argument("--per-job-lot", type=int, default=DEFAULT_IDS_PER_JOB["LOT"])

    ap.add_argument("--fallback-groups", type=int, default=1, help="groupsが取れない時の最終fallback")
    ap.add_argument("--debug", action="store_true", help="診断ログを出す")
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

    lots_by_edate, wafers_by_edate, ids_info = read_ids_counts_by_edate(
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

    data_types = list_data_types_from_specs()
    if not data_types:
        # 最悪 runs から推定（debug時のみ詳細）
        dts = set()
        for r in rows_all:
            if r.get("snapshot") != snap:
                continue
            jk = r.get("job_key")
            if jk:
                dt = parse_job_key(jk).get("data_type")
                if dt:
                    dts.add(dt)
        data_types = sorted(dts)
        if args.debug:
            print(f"[DEBUG] meta/specs not found -> data_types inferred from runs: {len(data_types)}")

    cls_by_dt = infer_cls_by_data_type_from_runs(rows_all, snap)
    ids_per_job = {"ROW": args.per_job_row, "WAFER": args.per_job_wafer, "LOT": args.per_job_lot}

    planned_jobs, planned_by_dt, planned_by_cls = calc_planned_jobs(
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

    # --- Core output (通常時) ---
    print(f"snapshot: {snap}")
    print(f"planned e_dates: {planned_edates}")
    print(f"planned lots:   {planned_lots}")
    print(f"planned wafers: {planned_wafers}")
    print(f"planned data_types: {len(data_types)}")
    print(f"TOTAL planned jobs: {planned_jobs}")

    if planned_jobs > 0:
        done_ratio = done / planned_jobs
        seen_ratio = seen / planned_jobs
        print("\nProgress:")
        print(f"  done (parquet_ok): {done:7d}/{planned_jobs:7d} [{fmt_bar(done_ratio)}] {done_ratio*100:5.1f}%")
        print(f"  seen (attempted):  {seen:7d}/{planned_jobs:7d} [{fmt_bar(seen_ratio)}] {seen_ratio*100:5.1f}%")
        print(f"  raw_only:          {raw_only:7d}")
    else:
        print("\n[ERROR] planned_jobs=0 です。")
        print("  ids.csv の LOT_ID/E_DATE が空扱いになっている、または groups 推定が全滅している可能性があります。")
        print("  `--debug` を付けて ids.csv の header_norm / rows_with_both / dt別groups を確認してください。")

    # --- Debug output (debug時のみ) ---
    if args.debug:
        print("\n[DEBUG] paths")
        print(f"  ROOT={ROOT}")
        print(f"  RUNS={RUNS} (exists={RUNS.exists()})")
        print(f"  IDS={ids_csv} (size={ids_csv.stat().st_size} bytes)")

        print("\n[DEBUG] ids.csv info")
        print(f"  encoding={ids_info.get('encoding')} delimiter={repr(ids_info.get('delimiter'))}")
        print(f"  header_norm={ids_info.get('header_norm', [])[:30]}")
        print(f"  total_rows={ids_info.get('total_rows')}")
        print(f"  rows_with_edate={ids_info.get('rows_with_edate')}")
        print(f"  rows_with_lot={ids_info.get('rows_with_lot')}")
        print(f"  rows_with_both={ids_info.get('rows_with_both')}")
        print(f"  has_lot_col={ids_info.get('has_lot_col')} has_wafer_col={ids_info.get('has_wafer_col')}")

        print("\n[DEBUG] run status breakdown")
        for k in sorted(status_cnt.keys()):
            if k not in ("parquet_ok", "raw_only"):
                print(f"  {k}: {status_cnt[k]}")

        print("\n[DEBUG] planned jobs by cls")
        for cls, v in sorted(planned_by_cls.items(), key=lambda x: -x[1]):
            print(f"  {cls:6s}: {v}")

        print("\n[DEBUG] planned jobs by data_type (top 30)")
        for dt, p in sorted(planned_by_dt.items(), key=lambda x: -x[1])[:30]:
            print(f"  {dt:12s} planned={p}")


if __name__ == "__main__":
    main()
