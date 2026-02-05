from __future__ import annotations

import os
import time
import gzip
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import polars as pl

# =========================
# 安全設定（最小）
# =========================
REQUIRED_KEYS_BY_DATASET = {
    "chips": ["wafer_id", "g_x", "g_y"],
    "wafers": ["wafer_id"],
}
KEY_NULL_RATE_MAX = 0.01
MAX_RETRIES = 3


@dataclass(frozen=True)
class BatchSpec:
    dt: str           # "YYYY-MM-DD"（管理用）
    group: str        # "0001"
    lot_ids: list[str]


@dataclass(frozen=True)
class TaskSpec:
    dataset: str                  # "chips" / "wafers"
    data_type: str                # "IQC" / "DS_CAT" / "DS_CHAR" / "REPORT"
    cat: str                      # 列セット名（カテゴリ名）
    columns: list[str]            # このタスクで取得される列（検証用）
    spec_sheet: Any               # webclient が要求する「データスペックシート内容」そのもの


def make_ingest_id() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


# =========================
# ここだけ差し替え（あなたのwebclient実装）
# =========================
def webclient_download_zip(*, spec_sheet: Any, data_type: str, lot_ids: list[str], out_zip_path: Path) -> None:
    """
    期待：ZIPを out_zip_path に保存（ZIP内CSV1つ）。
    """
    # 例：あなたの既存webclientがこういうI/Fだとして
    # webclient(spec_sheet, data_type, lot_ids, str(out_zip_path))
    raise NotImplementedError


# =========================
# パス規約（zipはtmpのみ）
# =========================
def tmp_task_dirs(root: Path, ingest_id: str, task: TaskSpec, batch: BatchSpec) -> dict[str, Path]:
    base = root / "tmp" / "ingest" / f"ingest_id={ingest_id}"
    key = Path(f"dataset={task.dataset}") / f"data_type={task.data_type}" / f"cat={task.cat}" / f"dt={batch.dt}" / f"group={batch.group}"
    return {
        "lock":      base / "locks" / f"dataset={task.dataset}__data_type={task.data_type}__cat={task.cat}__dt={batch.dt}__g={batch.group}.lock",
        "dl_dir":    base / "download" / key,
        "ex_dir":    base / "extract"  / key,
        "stage_pq":  base / "stage" / "bronze_parquet" / key,
        "stage_csv": base / "stage" / "bronze_csv" / key,
    }


def final_dirs(root: Path, ingest_id: str, task: TaskSpec, batch: BatchSpec) -> dict[str, Path]:
    key = Path(f"dataset={task.dataset}") / f"data_type={task.data_type}" / f"cat={task.cat}" / f"ingest_id={ingest_id}" / f"dt={batch.dt}" / f"group={batch.group}"
    return {
        "pq":  root / "data" / "bronze_parquet" / key,
        "csv": root / "data" / "bronze_csv" / key,
    }


# =========================
# 小物
# =========================
def acquire_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)


def release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def touch_success(dir_path: Path) -> None:
    (dir_path / "_SUCCESS").write_text("")


def is_done(dir_path: Path) -> bool:
    return (dir_path / "_SUCCESS").exists()


def ensure_no_partial_output(final_dir: Path) -> None:
    if final_dir.exists() and not is_done(final_dir):
        raise RuntimeError(f"中途半端な出力が存在（_SUCCESS無し）: {final_dir}")


def safe_move_dir(stage_dir: Path, final_dir: Path) -> None:
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        stage_dir.replace(final_dir)  # 同一FS想定
    except OSError:
        shutil.move(str(stage_dir), str(final_dir))  # 別FSフォールバック


def assert_required_keys_in_columns(dataset: str, columns: list[str]) -> None:
    required = REQUIRED_KEYS_BY_DATASET.get(dataset, [])
    missing = [k for k in required if k not in columns]
    if missing:
        raise ValueError(f"{dataset} の columns に必須キーが不足: missing={missing}")


# =========================
# ZIP展開（CSV1個前提）
# =========================
def extract_single_csv(zip_path: Path, out_csv_path: Path) -> None:
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError(f"ZIP内CSVが1個ではありません: {csv_names}")
        with zf.open(csv_names[0]) as src, open(out_csv_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


def read_csv_header_cols(csv_path: Path) -> list[str]:
    # ヘッダだけ読む（軽量）
    with open(csv_path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        header = f.readline().rstrip("\n\r")
    return [c.strip().strip('"') for c in header.split(",")]


def validate_csv_header(dataset: str, csv_cols: list[str], requested_cols: list[str]) -> None:
    required = REQUIRED_KEYS_BY_DATASET.get(dataset, [])
    missing_required = [k for k in required if k not in csv_cols]
    if missing_required:
        raise RuntimeError(f"CSVヘッダに必須キーが不足: missing={missing_required}")

    missing_req = [c for c in requested_cols if c not in csv_cols]
    if missing_req:
        raise RuntimeError(f"CSVヘッダに要求列が不足（API取り漏れ疑い）: missing={missing_req[:20]}...")


# =========================
# CSV -> Parquet + CSV(.gz) 変換
# =========================
def convert_csv_to_bronze_parts(
    csv_path: Path,
    stage_pq_dir: Path,
    stage_csv_dir: Path,
    *,
    batch_rows: int = 200_000,
    csv_compress: bool = True,
) -> dict:
    stage_pq_dir.mkdir(parents=True, exist_ok=True)
    stage_csv_dir.mkdir(parents=True, exist_ok=True)

    reader = pl.read_csv_batched(
        str(csv_path),
        batch_size=batch_rows,
        infer_schema_length=10_000,
        try_parse_dates=True,
    )

    part = 0
    total_rows = 0

    while True:
        batches = reader.next_batches(1)
        if not batches:
            break

        df = batches[0]
        total_rows += df.height

        pq_path = stage_pq_dir / f"part-{part:04d}.parquet"
        df.write_parquet(pq_path, compression="zstd")

        if csv_compress:
            out_csv = stage_csv_dir / f"part-{part:04d}.csv.gz"
            with gzip.open(out_csv, "wt", encoding="utf-8", newline="") as f:
                df.write_csv(f)
        else:
            out_csv = stage_csv_dir / f"part-{part:04d}.csv"
            df.write_csv(out_csv)

        part += 1

    if total_rows == 0:
        raise RuntimeError("変換結果が0行です（CSVが空 or 読み取り失敗）")

    return {"parts": int(part), "rows": int(total_rows)}


# =========================
# Parquet最小検証（キーnull率 + chipsはキー重複）
# =========================
def validate_parquet_keys(dataset: str, parquet_glob: str) -> dict:
    keys = REQUIRED_KEYS_BY_DATASET.get(dataset, [])
    lf = pl.scan_parquet(parquet_glob)

    for k in keys:
        if k not in lf.columns:
            raise RuntimeError(f"Parquetに必須キー列が存在しません: {k}")

    null_rates = lf.select([pl.col(k).is_null().mean().alias(f"{k}_null_rate") for k in keys]).collect().to_dicts()[0]

    dup_rows = None
    if dataset == "chips":
        n = lf.select(pl.len()).collect().item()
        nu = lf.select(pl.struct(keys).n_unique()).collect().item()
        dup_rows = int(n - nu)

    ok = all(float(null_rates[f"{k}_null_rate"]) <= KEY_NULL_RATE_MAX for k in keys)
    if dataset == "chips":
        ok = ok and (dup_rows == 0)

    return {"ok": bool(ok), **{k: float(v) for k, v in null_rates.items()}, "dup_key_rows": dup_rows}


def commit_stage_to_final(stage_dir: Path, final_dir: Path) -> None:
    if is_done(final_dir):
        return
    ensure_no_partial_output(final_dir)
    safe_move_dir(stage_dir, final_dir)


# =========================
# 1タスク実行
# =========================
def run_one_task(
    root: Path,
    ingest_id: str,
    *,
    task: TaskSpec,
    batch: BatchSpec,
    batch_rows: int = 200_000,
    csv_compress: bool = True,
) -> dict:
    assert_required_keys_in_columns(task.dataset, task.columns)

    finals = final_dirs(root, ingest_id, task, batch)
    if is_done(finals["pq"]) and is_done(finals["csv"]):
        return {"status": "skipped", "dataset": task.dataset, "data_type": task.data_type, "cat": task.cat, "dt": batch.dt, "group": batch.group}

    ensure_no_partial_output(finals["pq"])
    ensure_no_partial_output(finals["csv"])

    tmp = tmp_task_dirs(root, ingest_id, task, batch)
    lock = tmp["lock"]

    acquire_lock(lock)
    try:
        # download zip -> tmp（APIは data_type と spec_sheet で決まる）
        tmp["dl_dir"].mkdir(parents=True, exist_ok=True)
        zip_path = tmp["dl_dir"] / "raw.zip"

        for attempt in range(MAX_RETRIES):
            try:
                webclient_download_zip(
                    spec_sheet=task.spec_sheet,
                    data_type=task.data_type,
                    lot_ids=batch.lot_ids,
                    out_zip_path=zip_path,
                )
                if not zip_path.exists() or zip_path.stat().st_size == 0:
                    raise RuntimeError("ZIPが作成されていない/空です")
                break
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)

        # extract csv
        csv_path = tmp["ex_dir"] / "raw.csv"
        extract_single_csv(zip_path, csv_path)

        # CSVヘッダ検証（必須キー＋要求列）
        csv_cols = read_csv_header_cols(csv_path)
        validate_csv_header(task.dataset, csv_cols, task.columns)

        # stage掃除（同一タスクのみ）
        if tmp["stage_pq"].exists():
            shutil.rmtree(tmp["stage_pq"])
        if tmp["stage_csv"].exists():
            shutil.rmtree(tmp["stage_csv"])

        # convert -> stage
        stats = convert_csv_to_bronze_parts(
            csv_path,
            tmp["stage_pq"],
            tmp["stage_csv"],
            batch_rows=batch_rows,
            csv_compress=csv_compress,
        )

        # parquet検証
        v = validate_parquet_keys(task.dataset, str(tmp["stage_pq"] / "part-*.parquet"))
        if not v["ok"]:
            raise RuntimeError(f"キー検証に失敗: {v}")

        # stage success
        touch_success(tmp["stage_pq"])
        touch_success(tmp["stage_csv"])

        # commit
        commit_stage_to_final(tmp["stage_pq"], finals["pq"])
        commit_stage_to_final(tmp["stage_csv"], finals["csv"])

        # final success（念のため）
        touch_success(finals["pq"])
        touch_success(finals["csv"])

        return {
            "status": "success",
            "dataset": task.dataset, "data_type": task.data_type, "cat": task.cat,
            "dt": batch.dt, "group": batch.group,
            **stats,
            "validation": v,
        }
    finally:
        release_lock(lock)


# =========================
# 実行（カテゴリ並列）
# =========================
def run_bronze_ingest(
    root: Path,
    batches: list[BatchSpec],
    tasks: list[TaskSpec],
    *,
    max_workers: int = 4,
    batch_rows: int = 200_000,
    csv_compress: bool = True,
) -> None:
    ingest_id = make_ingest_id()

    for batch in batches:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(
                    run_one_task, root, ingest_id,
                    task=t, batch=batch,
                    batch_rows=batch_rows, csv_compress=csv_compress,
                )
                for t in tasks
            ]
            for fut in as_completed(futs):
                print(fut.result())
