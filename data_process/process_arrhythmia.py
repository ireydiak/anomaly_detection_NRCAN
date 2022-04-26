from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
import utils
import os
from typing import Tuple
from scipy.io import loadmat 
from datetime import datetime as dt
# import warnings

warnings.filterwarnings('ignore')


def clean_step(path_to_dataset: str, export_path: str, backup: bool = False) -> Tuple[pd.DataFrame, dict]:
    # Keep a trace of the cleaning step
    stats = defaultdict()
    stats["Dropped Columns"] = []
    stats["Dropped NaN Columns"] = []
    stats["NaN/INF Rows"] = 0

    # 1- Load file
    if not path_to_dataset.endswith(".mat"):
        raise Exception("process_arrhythmia can only process .mat files")
    mat = loadmat(path_to_dataset)
    X = mat['X']  # variable in mat file
    y = mat['y'].reshape(-1)
    # now make a data frame, setting the time stamps as the index
    df = pd.DataFrame(X, columns=None)

    # Remove leading and trailing spaces from columns names
    total_rows = len(df)
    stats["Total Rows"] = str(total_rows)
    stats["Total Features"] = len(df.columns)

    # 2- Start data cleaning
    # 2.1- Remove columns with unique values
    cols_uniq_vals = df.columns[df.nunique() <= 1].to_list()
    # df = df.drop(cols_uniq_vals, axis=1)
    stats["Unique Columns"] = " ".join([str(col) for col in cols_uniq_vals])
    stats["Dropped Columns"].extend(cols_uniq_vals)

    # 2.2- Drop columns with NaN or INF values
    # Transforming all invalid data in numerical columns to NaN
    num_cols = df.select_dtypes(exclude=["object", "category"]).columns.tolist()
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

    # Replacing INF values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    nan_cols = df.columns[(df.isna()).any()].tolist()
    stats["NaN Columns"] = " ".join(nan_cols)
    for col in nan_cols:
        nan_rows = (df[col].isna()).sum()
        if nan_rows >= 0.1 * len(df[col]):
            df = df.drop(col, axis=1)
            stats["Dropped NaN Columns"].append(col)
            stats["Dropped Columns"].append(col)
        else:
            stats["NaN/INF Rows"] += nan_rows
            df[col].dropna(inplace=True)

    assert df.isna().sum().sum() == 0

    deleted_rows = stats["NaN/INF Rows"]
    stats["Ratio"] = f"{(deleted_rows / total_rows):1.4f}" if deleted_rows > 0 else "0.0"
    stats["Final Features"] = str(len(df.columns))
    stats["Final Total Rows"] = str(len(df))
    for key, val in stats.items():
        if type(val) == list:
            stats[key] = " ".join(str(v) for v in val)
        elif type(val) != str:
            stats[key] = str(val)

    return df, y, stats


def normalize_step(df: pd.DataFrame, y: np.array, base_path: str, backup:bool = False, norm=True):
    print(f'Processing {len(df.columns)} features')
    # Split numerical and non-numerical columns
    num_cols = df.select_dtypes(exclude=["object", "category"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    # Categorical columns should have been removed already
    assert len(cat_cols) == 0
    # Normalize numerical data
    scaler = MinMaxScaler()
    # Select numerical columns with values in the range (0, 1)
    # This way we avoid normalizing values that are already between 0 and 1.
    to_scale = df[num_cols][(df[num_cols] < 0.0).any(axis=1) & (df[num_cols] > 1.0).any(axis=1)].columns
    print(f"Scaling {len(to_scale)} columns: " + ", ".join([str(col) for col in num_cols]))
    if norm:
        df[to_scale] = scaler.fit_transform(df[to_scale].values.astype(np.float64))
    # Merge normalized dataframe with labels
    X = np.concatenate(
        (df.values, y.reshape(-1, 1)),
        axis=1
    )
    if backup:
        normalized_fname = f'{base_path}/{utils.folder_struct["normalize_step"]}/arrhythmia_normalized.csv'
        df.to_csv(
            normalized_fname,
            sep=',', encoding='utf-8', index=False
        )
        print(f'Saved {normalized_fname}')
    compressed_fname = f'{base_path}/{utils.folder_struct["minify_step"]}/arrhythmia_normalized.npz'
    np.savez(compressed_fname, arrhythmia=X.astype(np.float64))
    print(f'Saved {compressed_fname}')


if __name__ == '__main__':
    # Assumes `path` points to the .mat file downloaded from http://odds.cs.stonybrook.edu/arrhythmia-dataset/ 
    path, export_path, backup, normlize_flag = utils.parse_args()
    # 0 - Prepare folder structure
    utils.prepare(export_path)
    path_to_clean = f"{export_path}/{utils.folder_struct['clean_step']}/arrhythmia_clean.csv"
    if os.path.isfile(path_to_clean):
        print("Clean file exists. Skipping cleaning step.")
        df = pd.read_csv(path_to_clean)
    else:
        # 1 - Clean the data (remove invalid rows and columns)
        df, y, clean_stats = clean_step(path, export_path, backup)
        # Save info about cleaning step
        utils.save_stats(export_path + '/usb_ids_info.csv', clean_stats)

    # 2 - Normalize numerical values and treat categorical values
    normalize_step(df, y, export_path,  norm=normlize_flag)
