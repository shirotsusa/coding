from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import polars as pl

# ========= あなたの環境に合わせてここを置き換え =========
# from your_module import FetchBR
class FetchBR:  # ダミー
    def __init__(self, path: str, input_dict: dict): ...
    def start(self) -> Any: ...
# =====================================================


# =========================
# 設定
# =========================
@dataclass(frozen=True)
class BronzeConfig:
    project_root: Path
    myid_ini_path: Path
    xlsm_path: Path

    # ids.csv: columns: e_date, lot_id
    ids_csv: Path

    # xlsmのシート名（= data_type 大カテゴリ）
    sheet_names: list[str]

    # FetchBR内部並列度 = input_dictのkey数
    max_jobs_per_call: int = 8

    # typeコード（API用）→ dataset（管理用）
    # 例: {"ROW":"chips","WAFER":"wafers","LOT":"lots"}
    type_to_dataset: dict[str, str] | None = None

    # typeごとの投入ID数（job粒度）
    # 例: {"ROW": 1000, "WAFER": 1000, "LOT": 50}
    ids_per_job_by_type: dict[str, int] | None = None

    # 1 lot あたりの wafer 数（lot_id末尾3桁を 001..N に置換）
    wafers_per_lot: int = 25

    # Polars
    infer_schema_length: int = 10_000

    # cls: search_patternsに必要。xlsm左3列に無いので、通常はシート名を使う。
    sheet_to_cls: dict[str, str] | None = None

    cleanup_tmp_files: bool = True


def now_snapshot() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


# =========================
# パス生成（確定版構成）
# =========================
def p_tmp_base(root: Path, snapshot: str) -> Path:
    return root / "tmp" / "ingest" / f"snapshot={snapshot}"

def p_tmp_download(root: Path, snapshot: str) -> Path:
    return p_tmp_base(root, snapshot) / "download"

def p_tmp_extract(root: Path, snapshot: str, job_key: str) -> Path:
    return p_tmp_base(root, snapshot) / "extract" / job_key

def p_bronze_raw_dir(root: Path, *, dataset: str, data_type: str, ope: str, e_date: str, snapshot: str) -> Path:
    return (
        root / "lake" / "bronze" / "raw" / "format=csv_gz"
        / f"dataset={dataset}" / f"data_type={data_type}" / f"ope={ope}"
        / f"e_date={e_date}" / f"snapshot={snapshot}"
    )

def p_bronze_pq_dir(root: Path, *, dataset: str, data_type: str, ope: str, e_date: str, snapshot: str) -> Path:
    return (
        root / "lake" / "bronze" / "data" / "format=parquet"
        / f"dataset={dataset}" / f"data_type={data_type}" / f"ope={ope}"
        / f"e_date={e_date}" / f"snapshot={snapshot}"
    )

def p_bronze_meta(root: Path) -> Path:
    return root / "lake" / "bronze" / "meta"


# =========================
# ユーティリティ
# =========================
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def write_success(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "_SUCCESS").write_text("", encoding="utf-8")

def gzip_copy(src: Path, dst_gz: Path, buf_size: int = 1024 * 1024) -> None:
    import gzip
    dst_gz.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fin, gzip.open(dst_gz, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=buf_size)

def is_zip_file(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == b"PK"

def extract_single_csv(zip_path: Path, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(csvs) != 1:
            raise RuntimeError(f"ZIP内CSVが1本ではありません: {csvs}")
        with zf.open(csvs[0]) as src, open(out_csv, "wb") as dst:
            shutil.copyfileobj(src, dst)

def csv_to_parquet(csv_path: Path, pq_path: Path, *, infer_schema_length: int) -> None:
    """
    - lazy + sink_parquet（あれば）優先
    - なければ collect → write_parquet
    - 文字コード不明なので候補を試す
    """
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    enc_candidates = ["utf8", "utf8-lossy", "cp932", "shift_jis"]

    last_err: Exception | None = None

    def _sink(lf: pl.LazyFrame) -> None:
        if hasattr(lf, "sink_parquet"):
            lf.sink_parquet(str(pq_path), compression="zstd")
        else:
            try:
                df = lf.collect(streaming=True)
            except TypeError:
                df = lf.collect()
            df.write_parquet(str(pq_path), compression="zstd")

    for enc in enc_candidates:
        try:
            lf = pl.scan_csv(
                str(csv_path),
                infer_schema_length=infer_schema_length,
                try_parse_dates=True,
                encoding=enc,
            )
            _sink(lf)
            return
        except TypeError:
            try:
                lf = pl.scan_csv(
                    str(csv_path),
                    infer_schema_length=infer_schema_length,
                    try_parse_dates=True,
                )
                _sink(lf)
                return
            except Exception as e:
                last_err = e
        except Exception as e:
            last_err = e

    raise RuntimeError(f"CSV→Parquet変換失敗: {csv_path}") from last_err

def chunks(seq: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def iter_dict_chunks(d: dict, max_items: int) -> Iterable[dict]:
    items = list(d.items())
    for i in range(0, len(items), max_items):
        yield dict(items[i:i+max_items])


# =========================
# IDsロード（e_date + lot_id から wafer_id を生成）
# =========================
@dataclass(frozen=True)
class IdPartition:
    lot_bases: list[str]    # LOTタイプに投入するlot_id（末尾3文字除去）
    wafer_ids: list[str]    # ROW/WAFERタイプに投入するwafer_id（lot_base + 001..N）

def _lot_base_from_lot_id(lot_id: str) -> str:
    if len(lot_id) < 3:
        raise ValueError(f"lot_idが短すぎます（末尾3文字置換が前提）: {lot_id}")
    return lot_id[:-3]

def _wafer_ids_from_lot_base(lot_base: str, n: int) -> list[str]:
    # 001..n を3桁ゼロ埋め
    return [f"{lot_base}{i:03d}" for i in range(1, n + 1)]

def load_ids_by_edate(ids_csv: Path, wafers_per_lot: int) -> dict[str, IdPartition]:
    df = pl.read_csv(str(ids_csv), columns=["e_date", "lot_id"])

    out: dict[str, IdPartition] = {}
    grouped = df.group_by("e_date").agg([pl.col("lot_id")]).to_dicts()

    for row in grouped:
        e_date = str(row["e_date"])
        lot_ids = [str(x) for x in row["lot_id"]]

        # lot_baseを出現順でユニーク化（同じlotが複数行あっても1回だけ扱う）
        lot_bases: list[str] = []
        seen = set()
        for lid in lot_ids:
            base = _lot_base_from_lot_id(lid)
            if base not in seen:
                seen.add(base)
                lot_bases.append(base)

        # wafer_idを全展開（lot_baseごとに 001..N）
        wafer_ids: list[str] = []
        for base in lot_bases:
            wafer_ids.extend(_wafer_ids_from_lot_base(base, wafers_per_lot))

        out[e_date] = IdPartition(
            lot_bases=lot_bases,
            wafer_ids=wafer_ids,
        )

    return out


# =========================
# Specロード（xlsm: 左3列=type, ope, cols / ENABLEで抽出 / (type,ope)でcols束ね）
# =========================
@dataclass(frozen=True)
class SpecGroup:
    type_code: str
    ope: str
    cols: list[str]

def _enable_to_bool(v: Any) -> bool:
    if v is True:
        return True
    if v is False or v is None:
        return False
    if pd.isna(v):
        return False
    if isinstance(v, (int, float)):
        return int(v) == 1
    s = str(v).strip().lower()
    return s in ("true", "1", "yes", "y", "t")

def load_spec_groups(xlsm_path: Path, sheet_name: str) -> list[SpecGroup]:
    df = pd.read_excel(xlsm_path, sheet_name=sheet_name)
    if "ENABLE" not in df.columns:
        raise ValueError(f"{sheet_name}: ENABLE列がありません")

    enabled = df[df["ENABLE"].map(_enable_to_bool)].copy()
    if enabled.empty:
        return []

    type_col = enabled.columns[0]
    ope_col  = enabled.columns[1]
    cols_col = enabled.columns[2]

    groups: list[SpecGroup] = []
    for (t, o), g in enabled.groupby([type_col, ope_col], dropna=False):
        cols_raw = [str(x) for x in g[cols_col].dropna().tolist()]
        cols = list(dict.fromkeys(cols_raw))  # 順序維持で重複除去
        if not cols:
            continue
        groups.append(SpecGroup(type_code=str(t), ope=str(o), cols=cols))
    return groups

def write_enabled_columns_meta(root: Path, sheet: str, groups: list[SpecGroup]) -> str:
    payload = {
        "sheet_name": sheet,
        "groups": [{"type": g.type_code, "ope": g.ope, "cols": g.cols} for g in groups],
    }
    path = p_bronze_meta(root) / "specs" / f"data_type={sheet}" / "enabled_columns.json"
    write_json_atomic(path, payload)
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


# =========================
# type→dataset、cls解決、job生成
# =========================
def resolve_dataset(cfg: BronzeConfig, type_code: str) -> str:
    if not cfg.type_to_dataset or type_code not in cfg.type_to_dataset:
        raise ValueError(f"type_to_dataset 未設定/未登録: type={type_code}")
    return cfg.type_to_dataset[type_code]

def resolve_cls(cfg: BronzeConfig, sheet_name: str) -> str:
    if cfg.sheet_to_cls and sheet_name in cfg.sheet_to_cls:
        return cfg.sheet_to_cls[sheet_name]
    return sheet_name

def ids_for_type(id_part: IdPartition, type_code: str) -> list[str]:
    # LOTはlot_base、それ以外はwafer_id
    if type_code == "LOT":
        return id_part.lot_bases
    return id_part.wafer_ids

def ids_per_job(cfg: BronzeConfig, type_code: str) -> int:
    if not cfg.ids_per_job_by_type or type_code not in cfg.ids_per_job_by_type:
        raise ValueError(f"ids_per_job_by_type 未設定/未登録: type={type_code}")
    return int(cfg.ids_per_job_by_type[type_code])

def build_input_dict(
    cfg: BronzeConfig,
    *,
    sheet_name: str,
    e_date: str,
    id_part: IdPartition,
    spec_groups: list[SpecGroup],
    tmp_download_dir: Path,
) -> dict[str, dict]:
    cls_value = resolve_cls(cfg, sheet_name)
    input_dict: dict[str, dict] = {}

    for g in spec_groups:
        dataset = resolve_dataset(cfg, g.type_code)
        id_list = ids_for_type(id_part, g.type_code)
        per_job = ids_per_job(cfg, g.type_code)

        for part_idx, id_chunk in enumerate(chunks(id_list, per_job)):
            job_key = (
                f"{dataset}__{sheet_name}"
                f"__ope={g.ope}"
                f"__type={g.type_code}"
                f"__e_date={e_date}"
                f"__part={part_idx:04d}"
            )
            out_file = tmp_download_dir / f"{job_key}.download"  # 中身でzip/csv判定

            input_dict[job_key] = {
                "lot_waf_list": id_chunk,
                "search_patterns": {
                    "type": g.type_code,  # ROW/WAFER/LOT
                    "cls": cls_value,
                    "ope": g.ope,
                    "cols": g.cols,
                },
                "output_dir": str(tmp_download_dir),
                "file_name": str(out_file),
            }

    return input_dict


# =========================
# FetchBR実行・回収・正規化
# =========================
def run_fetchbr(cfg: BronzeConfig, *, input_dict: dict) -> None:
    br = FetchBR(str(cfg.myid_ini_path), input_dict)
    br.start()

def normalize_one_job(
    cfg: BronzeConfig,
    *,
    snapshot: str,
    sheet_name: str,
    e_date: str,
    job_key: str,
    job: dict,
    spec_hash: str,
) -> dict:
    meta = p_bronze_meta(cfg.project_root)
    runs_path = meta / "runs" / "runs.jsonl"

    out_file = Path(job["file_name"])
    if (not out_file.exists()) or out_file.stat().st_size == 0:
        rec = {
            "ts": datetime.now().isoformat(),
            "status": "missing_output",
            "job_key": job_key,
            "sheet_name": sheet_name,
            "e_date": e_date,
            "snapshot": snapshot,
            "file_name": str(out_file),
        }
        append_jsonl(runs_path, rec)
        return rec

    tmp_ex = p_tmp_extract(cfg.project_root, snapshot, job_key)
    tmp_ex.mkdir(parents=True, exist_ok=True)

    extracted_csv: Path | None = None
    if is_zip_file(out_file):
        extracted_csv = tmp_ex / "raw.csv"
        extract_single_csv(out_file, extracted_csv)
        csv_path = extracted_csv
        source_format = "zip"
    else:
        csv_path = out_file
        source_format = "csv"

    sp = job["search_patterns"]
    dataset = resolve_dataset(cfg, sp["type"])
    ope = str(sp["ope"])

    raw_dir = p_bronze_raw_dir(cfg.project_root, dataset=dataset, data_type=sheet_name, ope=ope, e_date=e_date, snapshot=snapshot)
    pq_dir  = p_bronze_pq_dir(cfg.project_root,  dataset=dataset, data_type=sheet_name, ope=ope, e_date=e_date, snapshot=snapshot)

    part_str = job_key.split("__part=")[-1]
    raw_path = raw_dir / f"part-{part_str}.csv.gz"
    pq_path  = pq_dir  / f"part-{part_str}.parquet"

    raw_ok = True
    pq_ok = True
    raw_err = None
    pq_err = None

    try:
        gzip_copy(csv_path, raw_path)
    except Exception as e:
        raw_ok = False
        raw_err = repr(e)

    if raw_ok:
        try:
            csv_to_parquet(csv_path, pq_path, infer_schema_length=cfg.infer_schema_length)
        except Exception as e:
            pq_ok = False
            pq_err = repr(e)
    else:
        pq_ok = False

    rec = {
        "ts": datetime.now().isoformat(),
        "status": "success" if (raw_ok and pq_ok) else ("raw_only" if raw_ok else "failed"),
        "job_key": job_key,
        "sheet_name": sheet_name,
        "dataset": dataset,
        "ope": ope,
        "type_code": sp["type"],
        "e_date": e_date,
        "snapshot": snapshot,
        "source_format": source_format,
        "raw_path": str(raw_path) if raw_ok else None,
        "parquet_path": str(pq_path) if pq_ok else None,
        "spec_hash": spec_hash,
        "raw_error": raw_err,
        "parquet_error": pq_err,
    }
    append_jsonl(runs_path, rec)

    if cfg.cleanup_tmp_files:
        try:
            out_file.unlink(missing_ok=True)
        except Exception:
            pass
        if extracted_csv is not None:
            try:
                extracted_csv.unlink(missing_ok=True)
            except Exception:
                pass

    return rec

def update_catalog(cfg: BronzeConfig, records: list[dict]) -> None:
    meta = p_bronze_meta(cfg.project_root)
    catalog_path = meta / "catalog" / "catalog.parquet"

    rows = []
    for r in records:
        if r["status"] == "missing_output":
            continue
        rows.append({
            "layer": "bronze",
            "dataset": r.get("dataset"),
            "data_type": r.get("sheet_name"),
            "ope": r.get("ope"),
            "type_code": r.get("type_code"),
            "e_date": r.get("e_date"),
            "snapshot": r.get("snapshot"),
            "raw_path": r.get("raw_path"),
            "parquet_path": r.get("parquet_path"),
            "spec_hash": r.get("spec_hash"),
            "status": r.get("status"),
            "created_at": r.get("ts"),
        })

    if not rows:
        return

    new_df = pl.DataFrame(rows)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    if catalog_path.exists():
        old = pl.read_parquet(str(catalog_path))
        merged = pl.concat([old, new_df], how="vertical_relaxed")
    else:
        merged = new_df

    tmp = catalog_path.with_suffix(".tmp.parquet")
    merged.write_parquet(str(tmp), compression="zstd")
    tmp.replace(catalog_path)

def write_partition_meta(cfg: BronzeConfig, *, sheet_name: str, e_date: str, snapshot: str, id_part: IdPartition) -> None:
    meta_path = (
        p_bronze_meta(cfg.project_root) / "partitions"
        / f"data_type={sheet_name}" / f"e_date={e_date}" / f"snapshot={snapshot}"
        / "lot_ids.json"
    )
    write_json_atomic(meta_path, {
        "e_date": e_date,
        "n_lots": len(id_part.lot_bases),
        "n_wafers": len(id_part.wafer_ids),
        "wafers_per_lot": cfg.wafers_per_lot,
        "lot_ids": id_part.lot_bases,
    })

def write_lineage_meta(cfg: BronzeConfig, *, sheet_name: str, e_date: str, snapshot: str, spec_hash: str) -> None:
    meta_path = (
        p_bronze_meta(cfg.project_root) / "lineage"
        / f"data_type={sheet_name}" / f"e_date={e_date}" / f"snapshot={snapshot}"
        / "lineage.json"
    )
    write_json_atomic(meta_path, {
        "snapshot": snapshot,
        "data_type": sheet_name,
        "e_date": e_date,
        "e_date_definition": "lot_level",
        "spec_hash": spec_hash,
        "xlsm_path": str(cfg.xlsm_path),
        "ids_csv": str(cfg.ids_csv),
        "created_at": datetime.now().isoformat(),
    })


# =========================
# _SUCCESS 判定（欠落があれば絶対に付けない）
# =========================
def _parse_job_key(job_key: str) -> tuple[str, str, str, str, str]:
    parts = job_key.split("__")
    if len(parts) < 6:
        raise ValueError(f"job_keyの形式が想定外: {job_key}")

    dataset = parts[0]
    sheet = parts[1]
    ope = next(p.split("=", 1)[1] for p in parts if p.startswith("ope="))
    e_date = next(p.split("=", 1)[1] for p in parts if p.startswith("e_date="))
    part = next(p.split("=", 1)[1] for p in parts if p.startswith("part="))
    return dataset, sheet, ope, e_date, part

def finalize_success_markers(
    cfg: BronzeConfig,
    *,
    snapshot: str,
    expected_job_keys: list[str],
    job_records: list[dict],
) -> None:
    ok = {r["job_key"] for r in job_records if r.get("parquet_path")}

    exp_by_dir: dict[tuple[str, str], set[str]] = {}
    ok_by_dir: dict[tuple[str, str], set[str]] = {}

    for k in expected_job_keys:
        dataset, sheet, ope, e_date, _ = _parse_job_key(k)
        raw_dir = str(p_bronze_raw_dir(cfg.project_root, dataset=dataset, data_type=sheet, ope=ope, e_date=e_date, snapshot=snapshot))
        pq_dir  = str(p_bronze_pq_dir(cfg.project_root,  dataset=dataset, data_type=sheet, ope=ope, e_date=e_date, snapshot=snapshot))
        dkey = (raw_dir, pq_dir)

        exp_by_dir.setdefault(dkey, set()).add(k)
        if k in ok:
            ok_by_dir.setdefault(dkey, set()).add(k)

    for (raw_dir, pq_dir), exp in exp_by_dir.items():
        okset = ok_by_dir.get((raw_dir, pq_dir), set())
        if exp == okset:
            write_success(Path(raw_dir))
            write_success(Path(pq_dir))


# =========================
# メイン：Bronze生成
# =========================
def run_bronze(cfg: BronzeConfig) -> None:
    if not cfg.type_to_dataset:
        raise ValueError("type_to_dataset が必要です（ROW/WAFER/LOT の対応）")
    if not cfg.ids_per_job_by_type:
        raise ValueError("ids_per_job_by_type が必要です（type別のchunkサイズ）")

    snapshot = now_snapshot()

    meta = p_bronze_meta(cfg.project_root)
    (meta / "runs").mkdir(parents=True, exist_ok=True)
    (meta / "catalog").mkdir(parents=True, exist_ok=True)

    tmp_dl = p_tmp_download(cfg.project_root, snapshot)
    tmp_dl.mkdir(parents=True, exist_ok=True)

    ids_by_edate = load_ids_by_edate(cfg.ids_csv, cfg.wafers_per_lot)

    for sheet in cfg.sheet_names:
        spec_groups = load_spec_groups(cfg.xlsm_path, sheet)
        if not spec_groups:
            continue

        spec_hash = write_enabled_columns_meta(cfg.project_root, sheet, spec_groups)

        for e_date, id_part in ids_by_edate.items():
            write_partition_meta(cfg, sheet_name=sheet, e_date=e_date, snapshot=snapshot, id_part=id_part)
            write_lineage_meta(cfg, sheet_name=sheet, e_date=e_date, snapshot=snapshot, spec_hash=spec_hash)

            all_jobs = build_input_dict(
                cfg,
                sheet_name=sheet,
                e_date=e_date,
                id_part=id_part,
                spec_groups=spec_groups,
                tmp_download_dir=tmp_dl,
            )
            expected_keys = list(all_jobs.keys())
            all_records: list[dict] = []

            for wave_idx, wave in enumerate(iter_dict_chunks(all_jobs, cfg.max_jobs_per_call)):
                try:
                    run_fetchbr(cfg, input_dict=wave)
                except Exception as e:
                    append_jsonl(meta / "runs" / "runs.jsonl", {
                        "ts": datetime.now().isoformat(),
                        "status": "fetchbr_exception",
                        "sheet_name": sheet,
                        "e_date": e_date,
                        "snapshot": snapshot,
                        "wave_idx": wave_idx,
                        "error": repr(e),
                    })

                for job_key, job in wave.items():
                    rec = normalize_one_job(
                        cfg,
                        snapshot=snapshot,
                        sheet_name=sheet,
                        e_date=e_date,
                        job_key=job_key,
                        job=job,
                        spec_hash=spec_hash,
                    )
                    all_records.append(rec)

            update_catalog(cfg, all_records)
            finalize_success_markers(cfg, snapshot=snapshot, expected_job_keys=expected_keys, job_records=all_records)

    if cfg.cleanup_tmp_files:
        base = p_tmp_base(cfg.project_root, snapshot)
        extract_dir = base / "extract"
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)


if __name__ == "__main__":
    cfg = BronzeConfig(
        project_root=Path(".").resolve(),
        myid_ini_path=Path("config/myId.ini"),
        xlsm_path=Path("config/specs/data_spec.xlsm"),
        ids_csv=Path("config/ids.csv"),  # e_date, lot_id のみ
        sheet_names=["IQC", "DS_CAT", "DS_CHAR", "REPORT"],
        max_jobs_per_call=8,
        type_to_dataset={
            "ROW": "chips",
            "WAFER": "wafers",
            "LOT": "lots",
        },
        ids_per_job_by_type={
            "ROW": 1000,
            "WAFER": 1000,
            "LOT": 50,
        },
        wafers_per_lot=25,
        sheet_to_cls=None,
        cleanup_tmp_files=True,
    )
    run_bronze(cfg)
