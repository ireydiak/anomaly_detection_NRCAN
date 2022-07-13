from typing import Tuple
import warnings
import utils
import os
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings('ignore')

NORMAL_LABEL = 0
ANORMAL_LABEL = 1


def merge_step(path_to_files: str) -> Tuple[pd.DataFrame, dict]:
    total_rows = deleted_rows = 0
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
    stats["anomaly_ratio"] = "{:2.4f}".format((df["Label"] != "BENIGN").sum() / len(df))

    return pd.concat(chunks), stats


def clean_step(df: pd.DataFrame, stats: dict):
    # Group DoS attacks
    mask = df["Label"].str.startswith("DoS")
    df.loc[mask, "Label"] = "DoS"
    # Group Web attacks
    mask = df["Label"].str.startswith("Web Attack")
    df.loc[mask, "Label"] = "Web Attack"
    # Rename attacks to match the labels of IDS2018
    # Rename BENIGN to Benign
    mask = df["Label"].str.match("BENIGN")
    df.loc[mask, "Label"] = "Benign"
    # Rename FTP-Patator to FTP-BruteForce
    mask = df["Label"].str.match("FTP-Patator")
    df.loc[mask, "Label"] = "FTP-BruteForce"
    # Rename SSH-Patator to SSH-Bruteforce
    mask = df["Label"].str.match("SSH-Patator")
    df.loc[mask, "Label"] = "SSH-Bruteforce"

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

    # negative values
    num_cols = df.select_dtypes(exclude="object").columns
    mask = (df[num_cols] < 0).sum() > 0
    neg_cols = df[num_cols].columns[mask]
    stats["n_negative_cols"] = len(neg_cols)
    stats["negative_cols"] = ", ".join(neg_cols)
    print("Found {} columns with negative values: {}".format(len(neg_cols), neg_cols))
    neg_df = pd.DataFrame(
        pd.concat((
            (df[neg_cols] < 0).sum(),
            (df[neg_cols] < 0).sum() / len(df)
        ), axis=1)
    )
    neg_df.columns = ["Count", "Ratio"]
    neg_df = neg_df.sort_values("Count", ascending=False)

    num_cols = df.select_dtypes(exclude="object").columns
    mask = (df[num_cols] < 0).sum() > 0
    neg_cols = df[num_cols].columns[mask]
    stats["n_negative_cols"] = len(neg_cols)
    stats["negative_cols"] = ", ".join(neg_cols)
    print("Found {} columns with negative values: {}".format(len(neg_cols), neg_cols))
    # Drop `Init_Win_bytes_forward` and `Init_Win_bytes_backward` because too many of their values are equal to -1 which makes no sense.
    to_drop = neg_df[neg_df["Ratio"] > 0.01].index.tolist()
    df = df.drop(to_drop, axis=1)
    stats["n_dropped_cols"] += len(to_drop)
    stats["dropped_cols"] = stats["dropped_cols"] + ", ".join(to_drop)
    num_cols = df.select_dtypes(include=np.number).columns
    print("Dropped {} columns: {}".format(len(to_drop), to_drop))
    # When Flow Duration < 0, multiple columns are negative. Since these rows are only associated with BENIGN flows, we can drop them.
    n_dropped = (df["Flow Duration"] < 0).sum()
    stats["n_dropped_rows"] += n_dropped
    df = df[df["Flow Duration"] >= 0]
    print("Dropped {} rows".format(n_dropped))
    neg_cols_when_anomalies = df[num_cols].columns[
        (df[num_cols][((df[num_cols]).any(1)) & (df["Label"] != "BENIGN")] < 0).sum() > 0]
    t = neg_cols_when_anomalies
    to_drop = list(neg_cols_when_anomalies)
    stats["n_dropped_cols"] += len(neg_cols_when_anomalies)
    stats["dropped_cols"] = stats["dropped_cols"] + ", ".join(to_drop)
    df = df.drop(to_drop, axis=1)
    print("Dropped {} columns {}".format(len(to_drop), to_drop))

    #
    num_cols = df.select_dtypes(include=np.number).columns
    neg_cols_labels = df[(df[num_cols] < 0).any(1)]["Label"].unique()
    assert len(neg_cols_labels) == 1 and neg_cols_labels[0] == "BENIGN"
    idx_to_drop = df[(df[num_cols] < 0).any(1)].index
    n_dropped = len(idx_to_drop)
    stats["n_dropped_rows"] += n_dropped
    df = df.drop(idx_to_drop, axis=0)
    print("Dropped {} rows".format(n_dropped))
    assert (df[num_cols] < 0).any(1).sum() == 0, "There are still negative values"
    print("There are no more negative values")

    # Drop categorical attributes
    df = df.drop(["Destination Port"], axis=1)
    df["Category"] = df["Label"]
    df.loc[df["Label"] == "BENIGN", "Label"] = 0
    df.loc[df["Label"] != 0, "Label"] = 1

    return df, stats


def main():
    # Assumes `path` points to the location of the original CSV files.
    # `path` must only contain CSV files and not other file types such as folders.
    path, export_path, _ = utils.parse_args()

    # 1 - Clean the data (remove invalid rows and columns)
    df, stats = merge_step(path)
    df, stats = clean_step(df, stats)

    # Save info about cleaning step
    utils.save_stats(export_path + "/cicids2018_info.csv", stats)
    # Save final dataframe
    df.to_csv(export_path + "/ids2017.csv")


if __name__ == '__main__':
    main()
