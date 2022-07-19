from typing import Tuple
import warnings
import utils
import os
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings('ignore')

NORMAL_LABEL = 0
NORMAL_CAT = "Benign"
ANORMAL_LABEL = 1


def merge_step(path_to_files: str) -> Tuple[pd.DataFrame, dict]:
    chunks, chunk = [], None
    stats = defaultdict()
    df = pd.DataFrame()

    for f in os.listdir(path_to_files):
        chunk = pd.read_csv(os.path.join(path_to_files, f))
        chunk.columns = chunk.columns.str.strip()
        df = pd.concat((df, chunk))
        print(f)
    stats["dropped_cols"] = ""
    stats["n_dropped_cols"] = 0
    stats["n_dropped_rows"] = 0
    stats["n_instances"] = len(df)
    stats["n_features"] = df.shape[1] - 1
    stats["anomaly_ratio"] = "{:2.4f}".format(
        (df["Label"] != "BENIGN").sum() / len(df)
    )

    return df, stats


def uniformize_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Group DoS attacks
    mask = df["Label"].str.startswith("DoS")
    df.loc[mask, "Label"] = "DoS"
    # Group Web attacks
    mask = df["Label"].str.startswith("Web Attack")
    df.loc[mask, "Label"] = "Web Attack"
    # Rename attacks to match the labels of IDS2018
    # Rename BENIGN to Benign
    mask = df["Label"].str.match("BENIGN")
    df.loc[mask, "Label"] = NORMAL_CAT
    # Rename FTP-Patator to FTP-BruteForce
    mask = df["Label"].str.match("FTP-Patator")
    df.loc[mask, "Label"] = "FTP-BruteForce"
    # Rename SSH-Patator to SSH-Bruteforce
    mask = df["Label"].str.match("SSH-Patator")
    df.loc[mask, "Label"] = "SSH-Bruteforce"
    return df


def clean_uniq(df: pd.DataFrame, stats: dict) -> Tuple[pd.DataFrame, dict]:
    # unique values
    uniq_cols = df.columns[df.nunique() <= 1].tolist()
    stats["n_unique_cols"] = len(uniq_cols)
    if uniq_cols:
        print("Found {} columns with unique values: {}".format(len(uniq_cols), uniq_cols))
        stats["unique_cols"] = ", ".join([str(col) for col in uniq_cols])
        df.drop(uniq_cols, axis=1, inplace=True)
        stats["n_dropped_cols"] += len(uniq_cols)
        uniq_cols = df.columns[df.nunique() <= 1].tolist()
    assert len(uniq_cols) == 0, "Found {} columns with unique values: {}".format(len(uniq_cols), uniq_cols)
    print("Columns are valid with more than one distinct value")
    return df, stats


def clean_invalid(df: pd.DataFrame, stats: dict) -> Tuple[pd.DataFrame, dict]:
    # nan values
    # Replacing INF values with NaN
    df = df.replace([-np.inf, np.inf], np.nan)
    nan_cols = df.columns[df.isna().sum() > 0].tolist()
    stats["n_nan_cols"] = len(nan_cols)
    if nan_cols:
        stats["nan_cols"] = ", ".join([str(col) for col in nan_cols])
    print("Found NaN columns: {}".format(nan_cols))

    # replace invalid values
    ratio = df[nan_cols].isna().sum()[0] / len(df)
    df = df.fillna(0)
    print("Replaced {:2.4f}% of original data".format(ratio))
    remaining_nans = df.isna().sum().sum()
    assert remaining_nans == 0, "There are still {} NaN values".format(remaining_nans)

    return df, stats


def clean_negative(df: pd.DataFrame, stats: dict) -> Tuple[pd.DataFrame, dict]:
    n_anom_before = (df["Label"] != NORMAL_CAT).sum()
    # select numerical columns
    num_cols = df.select_dtypes(exclude="object").columns
    # create mask for negative values on numerical columns
    mask = (df[num_cols] < 0).sum() > 0
    # select the numerical columns with negative values
    neg_cols = df[num_cols].columns[mask]
    stats["n_negative_cols"] = len(neg_cols)
    stats["negative_cols"] = ", ".join(neg_cols)
    print("Found {} columns with negative values: {}".format(len(neg_cols), neg_cols))
    # Drop `Init_Win_bytes_forward` and `Init_Win_bytes_backward` because too many of their values are equal to -1 which makes no sense.
    # to_drop = ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]
    # stats["n_dropped_cols"] += len(to_drop)
    # stats["dropped_cols"] = stats["dropped_cols"] + ", ".join(to_drop)
    # df = df.drop(to_drop, axis=1)
    # print("Dropped {} columns (negative values): {}".format(len(to_drop), to_drop))
    # When Flow Duration < 0, multiple columns are negative. Since these rows are only associated with BENIGN flows, we can drop them.
    # n_dropped = (df["Flow Duration"] < 0).sum()
    # stats["n_dropped_rows"] += n_dropped
    # df = df[df["Flow Duration"] >= 0]
    # print("Dropped {} rows (negative values)".format(n_dropped))
    # remove columns with negative values and associated with attacks
    num_cols = df.select_dtypes(exclude="object").columns
    neg_cols_when_anomalies = df[num_cols].columns[
        (df[num_cols][((df[num_cols]).any(1)) & (df["Label"] != NORMAL_CAT)] < 0).sum() > 0
    ]
    to_drop = list(neg_cols_when_anomalies)
    stats["n_dropped_cols"] += len(neg_cols_when_anomalies)
    stats["dropped_cols"] = stats["dropped_cols"] + ", ".join(to_drop)
    df = df.drop(to_drop, axis=1)
    print("Dropped {} columns {} (negative values)".format(len(to_drop), to_drop))
    # remove remaining negative rows exclusively associated with `Benign` traffic
    df = df.reset_index()
    num_cols = df.select_dtypes(include=np.number).columns
    idx_to_drop = df[(df[num_cols] < 0).any(1)].index
    # weird hack to go around an annoying index behavior from pandas
    # selecting the index from the subset `num_cols` includes anomalies on the complete dataframe
    # hence, to avoid deleting attacks, we compute the intersection between normal data and remaining negative values
    idx_to_drop = list(set(df[(df.Label == NORMAL_CAT)].index) & set(idx_to_drop))
    n_dropped = len(idx_to_drop)
    stats["n_dropped_rows"] += n_dropped
    df = df.drop(idx_to_drop, axis=0)
    print("Dropped {} rows".format(n_dropped))
    assert (df[num_cols] < 0).any(1).sum() == 0, "There are still negative values"
    print("There are no more negative values")
    n_anom_after = (df["Label"] != NORMAL_CAT).sum()
    assert n_anom_before == n_anom_after, "dropped {} anomalies, aborting".format(n_anom_before - n_anom_after)
    return df, stats


def clean_step(df: pd.DataFrame, stats: dict):
    # 1- uniformize anomaly labels
    df = uniformize_labels(df)
    # 2- drop categorical attributes
    df = df.drop(["Destination Port"], axis=1)
    # 3- remove columns with unique values
    df, stats = clean_uniq(df, stats)
    # 4- manage NaN/INF and other invalid values
    df, stats = clean_invalid(df, stats)

    # negative values
    df, stats = clean_negative(df, stats)

    # Keep the full-labels aside before "binarizing" them
    df["Category"] = df["Label"]
    # Convert labels to binary labels
    df.loc[df["Label"] == NORMAL_CAT, "Label"] = 0
    df.loc[df["Label"] != 0, "Label"] = 1

    return df, stats


def main():
    # Assumes `path` points to the location of the original CSV files.
    # `path` must only contain CSV files and not other file types such as folders.
    path, export_path, _ = utils.parse_args()

    # 1- Merge the different CSV files into one dataframe
    df, stats = merge_step(path)
    # 2- Clean the data (remove invalid rows and columns, etc.)
    df, stats = clean_step(df, stats)

    # Save info about cleaning step
    utils.save_stats(
        os.path.join(export_path, "ids2017_info.csv"), stats
    )
    # Save final dataframe
    df.to_csv(
        os.path.join(export_path, "ids2017.csv")
    )


if __name__ == '__main__':
    main()
