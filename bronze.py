
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
    dt: str           # "YYYY-MM-DD"（管理用。APIには渡さない）
    group: str        # "0001"
    lot_ids: list[str]


@dataclass(frozen=True)
class TaskSpec:
    dataset: str                  # "chips" / "wafers"
    data_type: str                # "IQC" / "DS_CAT" / "DS_CHAR" / "REPORT"
    cat: str                      # 列セット名（カテゴリ）
    columns: list[str]            # 期待列（ヘッダ検証用、最低限キー列は必須）
    spec_sheet: Any               # webclient が要求する「データスペックシート内容」


def make_ingest_id() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


# =========================
# ここをあなたのwebclient呼び出しに差し替え
# 仕様：out_dir を渡すと、ZIPの出力パス(str)が返る
# =========================
def webclient_download_zip_return_path(
    *,
    spec_sheet: Any,
    data_type: str,
    lot_ids: list[str],
    out_dir: Path,
) -> str:
    """
    ここを差し替え：
      returned_path_str = webclient(spec_sheet, data_type, lot_ids, str(out_dir))
    """
    raise NotImplementedError


# =========================
# パス規約（ZIPはtmp、出力はdata）
# =========================
def tmp_task_dirs(root: Path, ingest_id: str, task: TaskSpec, batch: BatchSpec) -> dict[str, Path]:
    base = root / "tmp" / "ingest" / f"ingest_id={ingest_id}"
    key = (
        Path(f"dataset={task.dataset}")
        / f"data_type={task.data_type}"
        / f"cat={task.cat}"
        / f"dt={batch.dt}"
        / f"group={batch.group}"
    )
    return {
        "lock": base / "locks" / f"dataset={task.dataset}__data_type={task.data_type}__cat={task.cat}__dt={batch.dt}__g={batch.group}.lock",
        "dl_dir": base / "download" / key,
        "ex_dir": base / "extract" / key,
        "stage_pq": base / "stage" / "bronze_parquet" / key,
        "stage_csv": base / "stage" / "bronze_csv" / key,
    }


def final_dirs(root: Path, ingest_id: str, task: TaskSpec, batch: BatchSpec) -> dict[str, Path]:
    key = (
        Path(f"dataset={task.dataset}")
        / f"data_type={task.data_type}"
        / f"cat={task.cat}"
        / f"ingest_id={ingest_id}"
        / f"dt={batch.dt}"
        / f"group={batch.group}"
    )
    return {
        "pq": root / "data" / "bronze_parquet" / key,
        "csv": root / "data" / "bronze_csv" / key,
    }


# =========================
# ユーティリティ
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
    # _SUCCESS無しのfinalが残っているなら、汚染回避のため停止
    if final_dir.exists() and not is_done(final_dir):
        raise RuntimeError(f"中途半端な出力が存在（_SUCCESS無し）: {final_dir}")


def safe_move_dir(stage_dir: Path, final_dir: Path) -> None:
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        stage_dir.replace(final_dir)  # 同一FSならrename
    except OSError:
        shutil.move(str(stage_dir), str(final_dir))  # 別FSフォールバック


def assert_required_keys_in_columns(dataset: str, columns: list[str]) -> None:
    required = REQUIRED_KEYS_BY_DATASET.get(dataset, [])
    missing = [k for k in required if k not in columns]
    if missing:
        raise ValueError(f"{dataset} の columns に必須キーが不足: missing={missing}")


def _is_under(child: Path, parent: Path) -> bool:
    child_r = child.resolve()
    parent_r = parent.resolve()
    return os.path.commonpath([str(child_r)]) == os.path.commonpath([str(child_r), str(parent_r)])


# =========================
# webclient wrapper（out_dir受け取り & return-path対応）
# =========================
def webclient_download_zip_to_dir(*, task: TaskSpec, lot_ids: list[str], out_dir: Path) -> Path:
    """
    webclient は out_dir を受け取り、ZIPの出力パス(str)を返す。
    戻り値が (フルパス) / (ファイル名のみ) どちらでも対応。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    returned = webclient_download_zip_return_path(
        spec_sheet=task.spec_sheet,
        data_type=task.data_type,
        lot_ids=lot_ids,
        out_dir=out_dir,
    )

    if not returned:
        raise RuntimeError("webclient が出力パスを返しませんでした（None/空文字）")

    p = Path(returned)
    zip_path = (out_dir / p) if not p.is_absolute() else p

    if zip_path.suffix.lower() != ".zip":
        raise RuntimeError(f"webclient の戻り値が .zip ではありません: {zip_path}")

    if not _is_under(zip_path, out_dir):
        raise RuntimeError(f"webclient が out_dir 配下以外へ出力しています: {zip_path}")

    if (not zip_path.exists()) or zip_path.stat().st_size == 0:
        raise RuntimeError(f"ZIP が存在しない/空です: {zip_path}")

    return zip_path


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
    # ヘッダ1行だけ（軽い）
    with open(csv_path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        header = f.readline().rstrip("\n\r")
    return [c.strip().strip('"') for c in header.split(",")]


def validate_csv_header(dataset: str, csv_cols: list[str], expected_cols: list[str]) -> None:
    required = REQUIRED_KEYS_BY_DATASET.get(dataset, [])
    missing_required = [k for k in required if k not in csv_cols]
    if missing_required:
        raise RuntimeError(f"CSVヘッダに必須キーが不足: missing={missing_required}")

    # expected_cols は「取れるはず」の検証。多すぎる場合はコメントアウトしても良い。
    missing_expected = [c for c in expected_cols if c not in csv_cols]
    if missing_expected:
        raise RuntimeError(f"CSVヘッダに期待列が不足（取り漏れ疑い）: missing={missing_expected[:20]}...")


# =========================
# CSV保存（raw.csv -> csv.gz のストリーム圧縮）
# =========================
def gzip_copy(src_csv: Path, dst_csv_gz: Path, buf_size: int = 1024 * 1024) -> None:
    dst_csv_gz.parent.mkdir(parents=True, exist_ok=True)
    with open(src_csv, "rb") as fin, gzip.open(dst_csv_gz, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=buf_size)


# =========================
# CSV -> Parquet（優先：scan_csv→sink、失敗時：batched）
# =========================
def csv_to_parquet_streaming(csv_path: Path, out_parquet_path: Path) -> None:
    out_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # 可能なら streaming で一発出力
    try:
        lf = pl.scan_csv(
            str(csv_path),
            infer_schema_length=10_000,
            try_parse_dates=True,
        )
        lf.sink_parquet(str(out_parquet_path), compression="zstd")
        return
    except Exception:
        pass

    # フォールバック：read_csv_batched で小分けにして concat せずに書き出すのは難しいので、
    # 最小として「パートParquet」を作る（後でscan_parquetでまとめて扱える）
    if not hasattr(pl, "read_csv_batched"):
        raise RuntimeError("scan_csv→sink_parquet も read_csv_batched も使えません。Polars更新が必要です。")

    tmp_dir = out_parquet_path.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)

    reader = pl.read_csv_batched(
        str(csv_path),
        batch_size=200_000,
        infer_schema_length=10_000,
        try_parse_dates=True,
    )

    part = 0
    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
        df = batches[0]
        df.write_parquet(tmp_dir / f"part-{part:04d}.parquet", compression="zstd")
        part += 1

    if part == 0:
        raise RuntimeError("Parquet変換結果が0パートです（CSVが空 or 読み取り失敗）")


# =========================
# Parquet最小検証（キーnull率 + chipsはキー重複）
# =========================
def validate_parquet_keys(dataset: str, parquet_path_or_glob: str) -> dict:
    keys = REQUIRED_KEYS_BY_DATASET.get(dataset, [])
    lf = pl.scan_parquet(parquet_path_or_glob)

    # 列存在
    for k in keys:
        if k not in lf.columns:
            raise RuntimeError(f"Parquetに必須キー列が存在しません: {k}")

    null_rates = (
        lf.select([pl.col(k).is_null().mean().alias(f"{k}_null_rate") for k in keys])
        .collect()
        .to_dicts()[0]
    )

    dup_rows = None
    if dataset == "chips":
        # まず軽い方法（struct n_unique）を試す。版差で落ちたら group_by 方式にフォールバック。
        try:
            n = lf.select(pl.len()).collect().item()
            nu = lf.select(pl.struct(keys).n_unique()).collect().item()
            dup_rows = int(n - nu)
        except Exception:
            # group_by(keys).len() で len>1 を数える（少し重いが確実）
            dups = (
                lf.group_by(keys).len()
                .filter(pl.col("len") > 1)
                .select(pl.sum("len").alias("dup_rows"))
                .collect()
                .item()
            )
            dup_rows = int(dups or 0)

    ok = all(float(null_rates[f"{k}_null_rate"]) <= KEY_NULL_RATE_MAX for k in keys)
    if dataset == "chips":
        ok = ok and (dup_rows == 0)

    return {"ok": bool(ok), **{k: float(v) for k, v in null_rates.items()}, "dup_key_rows": dup_rows}


# =========================
# commit（stage -> final）
# =========================
def commit_stage_to_final(stage_dir: Path, final_dir: Path) -> None:
    if is_done(final_dir):
        return
    ensure_no_partial_output(final_dir)
    safe_move_dir(stage_dir, final_dir)


# =========================
# 1タスク実行（Bronze作成）
# =========================
def run_one_task(
    root: Path,
    ingest_id: str,
    *,
    task: TaskSpec,
    batch: BatchSpec,
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
        # --- download (zip) ---
        tmp["dl_dir"].mkdir(parents=True, exist_ok=True)

        # 誤認防止：タスク固有dirなので *.zip を掃除
        for z in tmp["dl_dir"].glob("*.zip"):
            z.unlink()

        for attempt in range(MAX_RETRIES):
            try:
                zip_path = webclient_download_zip_to_dir(task=task, lot_ids=batch.lot_ids, out_dir=tmp["dl_dir"])
                break
            except Exception:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)

        # --- extract csv（ZIP内CSV1個） ---
        csv_path = tmp["ex_dir"] / "raw.csv"
        if tmp["ex_dir"].exists():
            shutil.rmtree(tmp["ex_dir"])
        extract_single_csv(zip_path, csv_path)

        # --- header validate ---
        csv_cols = read_csv_header_cols(csv_path)
        validate_csv_header(task.dataset, csv_cols, task.columns)

        # --- stage cleanup ---
        if tmp["stage_pq"].exists():
            shutil.rmtree(tmp["stage_pq"])
        if tmp["stage_csv"].exists():
            shutil.rmtree(tmp["stage_csv"])
        tmp["stage_pq"].mkdir(parents=True, exist_ok=True)
        tmp["stage_csv"].mkdir(parents=True, exist_ok=True)

        # --- write bronze_csv (raw as csv.gz or csv) ---
        if csv_compress:
            out_csv = tmp["stage_csv"] / "data.csv.gz"
            gzip_copy(csv_path, out_csv)
        else:
            out_csv = tmp["stage_csv"] / "data.csv"
            shutil.copyfile(csv_path, out_csv)

        # --- write bronze_parquet ---
        # まずは単一ファイル出力を試す。失敗時は part-0000... 方式へフォールバック。
        out_parquet = tmp["stage_pq"] / "data.parquet"
        csv_to_parquet_streaming(csv_path, out_parquet)

        # --- validate parquet keys ---
        pq_target = str(out_parquet)
        if not out_parquet.exists():
            # フォールバックで part-*.parquet を作った可能性
            pq_target = str(tmp["stage_pq"] / "part-*.parquet")
        v = validate_parquet_keys(task.dataset, pq_target)
        if not v["ok"]:
            raise RuntimeError(f"キー検証に失敗: {v}")

        # --- stage success ---
        touch_success(tmp["stage_pq"])
        touch_success(tmp["stage_csv"])

        # --- commit ---
        commit_stage_to_final(tmp["stage_pq"], finals["pq"])
        commit_stage_to_final(tmp["stage_csv"], finals["csv"])

        # 念のためfinalにも_SUCCESS
        touch_success(finals["pq"])
        touch_success(finals["csv"])

        return {
            "status": "success",
            "dataset": task.dataset,
            "data_type": task.data_type,
            "cat": task.cat,
            "dt": batch.dt,
            "group": batch.group,
            "parquet": pq_target,
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
    csv_compress: bool = True,
) -> None:
    ingest_id = make_ingest_id()

    for batch in batches:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(
                    run_one_task,
                    root,
                    ingest_id,
                    task=t,
                    batch=batch,
                    csv_compress=csv_compress,
                )
                for t in tasks
            ]
            for fut in as_completed(futs):
                print(fut.result())


# =========================
# 使い方（例）
# =========================
if __name__ == "__main__":
    root = Path(".").resolve()

    batches = [
        BatchSpec(dt="2026-02-01", group="0001", lot_ids=["LOT001", "LOT002"]),
    ]

    # spec_sheet は xlsm から読み込んだ「データスペックシート内容」をそのまま入れる想定
    spec_sheet_defect = object()  # ここは実データに置換

    tasks = [
        TaskSpec(
            dataset="chips",
            data_type="IQC",
            cat="defect",
            columns=["wafer_id", "g_x", "g_y", "d1", "d2"],  # 期待列（最低限キーは必須）
            spec_sheet=spec_sheet_defect,
        ),
    ]

    run_bronze_ingest(
        root=root,
        batches=batches,
        tasks=tasks,
        max_workers=4,
        csv_compress=True,
    )
