from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
import utils
import os
from typing import Tuple, Any
from scipy.io import loadmat
from datetime import datetime as dt

# import warnings

warnings.filterwarnings('ignore')


# link to the dataset http://odds.cs.stonybrook.edu/thyroid-disease-dataset/

def clean_step(path_to_dataset: str, export_path: str, backup: bool = False) -> Tuple[Any, Any, defaultdict]:
    # Keep a trace of the cleaning step
    stats = defaultdict()
    stats["Dropped Columns"] = []
    stats["Dropped NaN Columns"] = []
    stats["NaN/INF Rows"] = 0

    # 1- Load file
    if not path_to_dataset.endswith(".mat"):
        raise Exception(f"{__file__} can only process .mat files")
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
    df = df.drop(cols_uniq_vals, axis=1)
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


def main():
    # Assumes `path` points to the ann-*.data
    path, export_path, backup = utils.parse_args()
    # 0 - Prepare folder structure
    utils.prepare(export_path)
    path_to_clean = f"{export_path}/{utils.folder_struct['clean_step']}/ann_thyroid.csv"

    # 1 - Clean the data (remove invalid rows and columns)
    df, y, clean_stats = clean_step(path, export_path, backup)
    # Save info about cleaning step
    utils.save_stats(export_path + '/ann_thyroid_info.csv', clean_stats)

    X = np.concatenate((
        df.to_numpy(),
        np.expand_dims(y, 1)
    ), axis=1)
    compressed_fname = f'{export_path}/{utils.folder_struct["minify_step"]}/thyroid'
    np.save(compressed_fname, thyroid=X.astype(np.float64))


if __name__ == '__main__':
    main()
