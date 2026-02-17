# bronze_extract_with_alias.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Mapping

import pandas as pd
import polars as pl


# =========================
# パス（make_bronze.py と同じ規約）
# =========================
def p_bronze_meta(root: Path) -> Path:
    return root / "lake" / "bronze" / "meta"

def p_catalog_parquet(root: Path) -> Path:
    return p_bronze_meta(root) / "catalog" / "catalog.parquet"

def p_partition_lot_ids_json(root: Path, *, data_type: str, e_date: str, snapshot: str) -> Path:
    return (
        p_bronze_meta(root) / "partitions"
        / f"data_type={data_type}" / f"e_date={e_date}" / f"snapshot={snapshot}"
        / "lot_ids.json"
    )


@dataclass(frozen=True)
class BronzeKey:
    dataset: str
    data_type: str   # xlsmのシート名（例: IQC, DS_CHAR...）
    ope: str
    cls: str


# =========================
# Catalog操作
# =========================
def load_catalog(project_root: Path) -> pl.DataFrame:
    path = p_catalog_parquet(project_root)
    if not path.exists():
        raise FileNotFoundError(f"catalog.parquet が見つかりません: {path}")
    df = pl.read_parquet(str(path))
    need = ["dataset", "data_type", "ope", "cls", "e_date", "snapshot", "parquet_path", "status"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"catalog.parquet に必要列がありません: {missing}\n実列: {df.columns}")
    return df

def list_partitions(cat: pl.DataFrame, *, only_success: bool = True, only_parquet: bool = True) -> pl.DataFrame:
    df = cat
    if only_success:
        df = df.filter(pl.col("status") == "success")
    if only_parquet:
        df = df.filter(pl.col("parquet_path").is_not_null())
    return (
        df.select(["dataset", "data_type", "ope", "cls", "e_date", "snapshot", "status"])
          .unique()
          .sort(["data_type", "e_date", "snapshot"])
    )

def latest_snapshot_for_date(cat: pl.DataFrame, *, key: BronzeKey, e_date: str, status: str = "success") -> str:
    df = (
        cat.filter(
            (pl.col("dataset") == key.dataset)
            & (pl.col("data_type") == key.data_type)
            & (pl.col("ope") == key.ope)
            & (pl.col("cls") == key.cls)
            & (pl.col("e_date") == e_date)
            & (pl.col("status") == status)
            & (pl.col("parquet_path").is_not_null())
        )
        .select(pl.col("snapshot").max())
    )
    v = df.item() if df.height else None
    if v is None:
        raise ValueError(f"該当なし: key={key}, e_date={e_date}, status={status}")
    return v

def parquet_paths_for(cat: pl.DataFrame, *, key: BronzeKey, e_date: str, snapshot: str, status: str = "success") -> list[str]:
    df = (
        cat.filter(
            (pl.col("dataset") == key.dataset)
            & (pl.col("data_type") == key.data_type)
            & (pl.col("ope") == key.ope)
            & (pl.col("cls") == key.cls)
            & (pl.col("e_date") == e_date)
            & (pl.col("snapshot") == snapshot)
            & (pl.col("status") == status)
            & (pl.col("parquet_path").is_not_null())
        )
        .select("parquet_path")
    )
    return df.to_series().to_list()

def autopick_latest_key_and_date(cat: pl.DataFrame) -> tuple[BronzeKey, str]:
    df = cat.filter((pl.col("status") == "success") & (pl.col("parquet_path").is_not_null()))
    if df.is_empty():
        raise ValueError("status=success かつ parquet_pathあり のレコードがありません")

    df2 = df.with_columns(pl.col("e_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("_d"))
    max_d = df2.select(pl.col("_d").max()).item()
    if max_d is None:
        max_e_date = df.select(pl.col("e_date").max()).item()
    else:
        max_e_date = df2.filter(pl.col("_d") == max_d).select(pl.col("e_date").first()).item()

    one = df.filter(pl.col("e_date") == max_e_date).select(["dataset", "data_type", "ope", "cls"]).unique().head(1)
    key = BronzeKey(dataset=one["dataset"][0], data_type=one["data_type"][0], ope=one["ope"][0], cls=one["cls"][0])
    return key, max_e_date


# =========================
# xlsm: (ope, cols) -> alias を読む
# =========================
def load_alias_map_from_xlsm(
    xlsm_path: Path,
    *,
    sheet_name: str,
    ope: str,
    ope_col: str = "ope",
    cols_col: str = "cols",
    alias_col: str = "alias",
    enabled_col: str | None = None,   # 例: "enabled" があるなら指定（Trueだけ採用）
) -> dict[str, str]:
    """
    xlsmの指定シートから、指定opeの (cols -> alias) を作る。
    想定: 1行が1カラム（cols 1つに alias 1つ）で、opeでグルーピングされている。
    """
    if not xlsm_path.exists():
        raise FileNotFoundError(f"xlsm が見つかりません: {xlsm_path}")

    # xlsmはopenpyxlで読める（マクロは無視される）
    df = pd.read_excel(xlsm_path, sheet_name=sheet_name, engine="openpyxl")

    for c in [ope_col, cols_col, alias_col]:
        if c not in df.columns:
            raise ValueError(f"xlsmシート({sheet_name})に列 {c!r} がありません。実列={list(df.columns)}")

    # 正規化
    d = df.copy()
    d[ope_col] = d[ope_col].astype(str).str.strip()
    d[cols_col] = d[cols_col].astype(str).str.strip()
    d[alias_col] = d[alias_col].astype(str).str.strip()

    # 空やnanっぽい値を除去
    d = d[(d[ope_col] != "") & (d[cols_col] != "") & (d[alias_col] != "")]
    d = d[~d[cols_col].str.lower().isin(["nan", "none"])]
    d = d[~d[alias_col].str.lower().isin(["nan", "none"])]

    # enabled列があるならTrueだけ
    if enabled_col is not None and enabled_col in d.columns:
        d = d[d[enabled_col].fillna(False).astype(bool)]

    # opeで絞る
    d = d[d[ope_col] == str(ope).strip()]
    if d.empty:
        return {}

    # cols->alias
    m = dict(zip(d[cols_col].tolist(), d[alias_col].tolist()))
    return m


def build_safe_rename_map(
    rename_map: Mapping[str, str],
    *,
    prefix: str | None = None,          # 例: f"{ope}__" を付けたいなら
    suffix_if_dup: str | None = None,   # alias重複時に付与（例: "__dup"）
) -> dict[str, str]:
    """
    aliasの重複/衝突が怖いときの安全策。
    """
    out: dict[str, str] = {}
    used: set[str] = set()

    for src, dst in rename_map.items():
        new = f"{prefix}{dst}" if prefix else dst
        if new in used and suffix_if_dup:
            k = 2
            cand = f"{new}{suffix_if_dup}{k}"
            while cand in used:
                k += 1
                cand = f"{new}{suffix_if_dup}{k}"
            new = cand
        used.add(new)
        out[src] = new

    return out


def apply_alias_rename(
    lf: pl.LazyFrame,
    rename_map: Mapping[str, str],
) -> pl.LazyFrame:
    """
    Polarsで列名をaliasへ変換。strict=False なので、存在しない列が混ざっても落ちない。
    """
    if not rename_map:
        return lf
    return lf.rename(dict(rename_map), strict=False)


# =========================
# meta/lot_ids.json
# =========================
def read_lot_ids_meta(project_root: Path, *, data_type: str, e_date: str, snapshot: str) -> dict:
    p = p_partition_lot_ids_json(project_root, data_type=data_type, e_date=e_date, snapshot=snapshot)
    if not p.exists():
        raise FileNotFoundError(f"lot_ids.json が見つかりません: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


# =========================
# 実行例
# =========================
if __name__ == "__main__":
    PROJECT_ROOT = Path(".")
    XLSM_PATH = Path("config/specs/data_spec.xlsm")  # make_bronze.py の例と同じ:contentReference[oaicite:1]{index=1}

    cat = load_catalog(PROJECT_ROOT)

    # (A) 何があるか
    dims = list_partitions(cat)
    print("\n=== Available partitions (sample) ===")
    print(dims.head(20))

    # (B) 対象を明示（Noneなら適当に1つ自動選択）
    KEY: BronzeKey | None = None
    E_DATE: str | None = None
    if KEY is None or E_DATE is None:
        KEY, E_DATE = autopick_latest_key_and_date(cat)
        print("\n=== Auto picked ===")
        print("KEY  :", KEY)
        print("E_DATE:", E_DATE)

    # (C) 最新snapshotのpathsをscan
    snap = latest_snapshot_for_date(cat, key=KEY, e_date=E_DATE)
    paths = parquet_paths_for(cat, key=KEY, e_date=E_DATE, snapshot=snap)
    lf = pl.scan_parquet(paths)

    # (D) xlsmのaliasでrename（このKEY.ope/KEY.data_typeに対応するものだけ）
    # NOTE: xlsm側の列名が ope/cols/alias じゃない場合は ope_col/cols_col/alias_col を変更
    raw_map = load_alias_map_from_xlsm(
        XLSM_PATH,
        sheet_name=KEY.data_type,
        ope=KEY.ope,
        ope_col="ope",
        cols_col="cols",
        alias_col="alias",
        enabled_col=None,   # enabled列があるなら "enabled" 等を入れる
    )

    # 重複や衝突が怖ければ prefix を付ける（例: ope別に区別したいとき）
    # safe_map = build_safe_rename_map(raw_map, prefix=f"{KEY.ope}__", suffix_if_dup="__dup")
    safe_map = build_safe_rename_map(raw_map, prefix=None, suffix_if_dup="__dup")

    lf = apply_alias_rename(lf, safe_map)

    # (E) 抽出（必要に応じて select/filter ）
    df = lf.collect(streaming=True)
    print("\n=== Extracted ===")
    print(df.shape)

    # (F) lot_ids meta（あれば）
    try:
        meta = read_lot_ids_meta(PROJECT_ROOT, data_type=KEY.data_type, e_date=E_DATE, snapshot=snap)
        print("\n=== lot_ids meta ===")
        print({k: meta.get(k) for k in ["e_date", "n_lots", "n_wafers", "wafers_per_lot"]})
    except FileNotFoundError:
        print("\n[INFO] lot_ids.json は未生成 or 該当パスにありません")
