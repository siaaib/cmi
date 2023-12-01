import shutil
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm import tqdm

from src.utils.common import trace

SERIES_SCHEMA = {
    "series_id": pl.Utf8,
    "step": pl.UInt32,
    "anglez": pl.Float32,
    "enmo": pl.Float32,
}


FEATURE_NAMES = [
    "anglez",
    "anglez_original",
    "anglez_sin",
    "anglez_cos",
    "enmo",
    "hour_sin",
    "hour_cos",
    "minute",
    "anglez_diff_rolling_med_65",
    "anglez_diff_rolling_mean_65",
    "anglez_diff_rolling_max_65",
    "anglez_diff_rolling_std_65",
    "enmo_diff_rolling_med_65",
    "enmo_diff_rolling_mean_65",
    "enmo_diff_rolling_max_65",
    "enmo_diff_rolling_std_65",
    "anglez_diff_rolling_med_33",
    "anglez_diff_rolling_mean_33",
    "anglez_diff_rolling_max_33",
    "anglez_diff_rolling_std_33",
    "enmo_diff_rolling_med_33",
    "enmo_diff_rolling_mean_33",
    "enmo_diff_rolling_max_33",
    "enmo_diff_rolling_std_33",
    "anglez_diff_rolling_med_17",
    "anglez_diff_rolling_mean_17",
    "anglez_diff_rolling_max_17",
    "anglez_diff_rolling_std_17",
    "enmo_diff_rolling_med_17",
    "enmo_diff_rolling_mean_17",
    "enmo_diff_rolling_max_17",
    "enmo_diff_rolling_std_17",
    "anglez_diff_1",
    "anglez_diff_2",
    "anglez_diff_4",
    "anglez_diff_8",
    "anglez_diff_16",
    "enmo_diff_1",
    "enmo_diff_2",
    "enmo_diff_4",
    "enmo_diff_8",
    "enmo_diff_16",
]

ANGLEZ_MEAN = -8.810476
ANGLEZ_STD = 35.521877
ENMO_MEAN = 0.041315
ENMO_STD = 0.101829


def to_coord(x: pl.Expr, max_: int, name: str) -> list[pl.Expr]:
    rad = 2 * np.pi * (x % max_) / max_
    x_sin = rad.sin()
    x_cos = rad.cos()

    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]


def to_rad_coord(x: pl.Expr, name: str) -> list[pl.Expr]:
    rad = x * np.pi / 180
    x_sin = rad.sin()
    x_cos = rad.cos()

    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]

def minute_is_important(x: pl.Expr) -> pl.Expr:
    result = (x % 15).cast(pl.Int8)
    return [result.alias("minute")]

def diff_rolling_feats(x: pl.Expr, window: int, name) -> pl.Expr:
    x_diff = x.diff(1).abs().fill_null(0)
    x_diff_rolling_med = x_diff.rolling_median(window, center=True).fill_null(0)
    x_diff_rolling_mean = x_diff.rolling_mean(window, center=True).fill_null(0)
    x_diff_rolling_std = x_diff.rolling_std(window, center=True).fill_null(0)
    x_diff_rolling_max = x_diff.rolling_max(window, center=True).fill_null(0)
    return [
        x_diff_rolling_med.alias(f"{name}_diff_rolling_med_{window}"),
        x_diff_rolling_mean.alias(f"{name}_diff_rolling_mean_{window}"),
        x_diff_rolling_max.alias(f"{name}_diff_rolling_max_{window}"),
        x_diff_rolling_std.alias(f"{name}_diff_rolling_std_{window}"),
    ]

def diff_feats(x: pl.Expr, name) -> pl.Expr:
    x_diff_1 = x.diff(1).abs().fill_null(0)
    x_diff_2 = x.diff(2).abs().fill_null(0)
    x_diff_4 = x.diff(4).abs().fill_null(0)
    x_diff_8 = x.diff(8).abs().fill_null(0)
    x_diff_16 = x.diff(16).abs().fill_null(0)

    return [
        x_diff_1.alias(f"{name}_diff_1"),
        x_diff_2.alias(f"{name}_diff_2"),
        x_diff_4.alias(f"{name}_diff_4"),
        x_diff_8.alias(f"{name}_diff_8"),
        x_diff_16.alias(f"{name}_diff_16")]

def add_feature(series_df: pl.DataFrame) -> pl.DataFrame:
    series_df = series_df.with_columns(
        *to_coord(pl.col("timestamp").dt.hour(), 24, "hour"),
        *minute_is_important(pl.col("timestamp").dt.minute()),
        *to_rad_coord(pl.col("anglez_original"), "anglez"),
        *diff_rolling_feats(pl.col("anglez"), 65, "anglez"),
        *diff_rolling_feats(pl.col("enmo"), 65, "enmo"),
        *diff_rolling_feats(pl.col("anglez"), 33, "anglez"),
        *diff_rolling_feats(pl.col("enmo"), 33, "enmo"),
        *diff_rolling_feats(pl.col("anglez"), 17, "anglez"),
        *diff_rolling_feats(pl.col("enmo"), 17, "enmo"),
        *diff_feats(pl.col("anglez"), "anglez"),
        *diff_feats(pl.col("enmo"), "enmo"),
    ).select("series_id", *FEATURE_NAMES)
    return series_df


def save_each_series(this_series_df: pl.DataFrame, columns: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col_name in columns:
        x = this_series_df.get_column(col_name).to_numpy(zero_copy_only=True)
        np.save(output_dir / f"{col_name}.npy", x)


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: DictConfig):
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.phase

    # ディレクトリが存在する場合は削除
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print(f"Removed {cfg.phase} dir: {processed_dir}")

    with trace("Load series"):
        # scan parquet
        if cfg.phase in ["train", "test"]:
            series_lf = pl.scan_parquet(
                Path(cfg.dir.data_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        elif cfg.phase == "dev":
            series_lf = pl.scan_parquet(
                Path(cfg.dir.processed_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        else:
            raise ValueError(f"Invalid phase: {cfg.phase}")

        # preprocess
        series_df = (
            series_lf.with_columns(
                pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z", time_zone="UTC"),
                pl.col("anglez").alias("anglez_original"),
                (pl.col("anglez") - ANGLEZ_MEAN) / ANGLEZ_STD,
                (pl.col("enmo") - ENMO_MEAN) / ENMO_STD,
            )
            .select(
                [
                    pl.col("series_id"),
                    pl.col("anglez"),
                    pl.col("anglez_original"),
                    pl.col("enmo"),
                    pl.col("timestamp"),
                ]
            )
            .collect(streaming=True)
            .sort(by=["series_id", "timestamp"])
        )
        n_unique = series_df.get_column("series_id").n_unique()
    with trace("Save features"):
        for series_id, this_series_df in tqdm(series_df.group_by("series_id"), total=n_unique):
            this_series_df = add_feature(this_series_df)

            series_dir = processed_dir / series_id  # type: ignore
            save_each_series(this_series_df, FEATURE_NAMES, series_dir)


if __name__ == "__main__":
    main()
