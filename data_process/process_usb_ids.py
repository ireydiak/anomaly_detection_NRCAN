from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
import warnings
import utils
import os
from typing import Tuple

warnings.filterwarnings('ignore')

rank_7_otf_7 = [
    'Flow IAT Max'
]

rank_6_otf_7 = [
    'Fwd Pkts/s',
    'Fwd Header Len',
    'Flow IAT Max',
    'Bwd Pkts/s',
    'Fwd Pkt Len Max'
]

rank_5_otf_7 = [
    'Fwd Pkts/s',
    'Fwd Header Len',
    'Fwd Pkt Len Mean',
    'Fwd IAT Tot',
    'Flow IAT Max',
    'Bwd Pkts/s',
    'Fwd Pkt Len Max'
]

rank_4_otf_7 = [
    'Fwd Pkt Len Mean', 'Fwd Pkt Len Max', 'Flow IAT Mean', 'TotLen Fwd Pkts', 'Bwd Pkts/s', 'Fwd Pkts/s',
    'Flow Byts/s', 'Fwd IAT Max', 'Fwd IAT Tot', 'Flow IAT Std', 'Flow IAT Max', 'Fwd Seg Size Min',
    'Flow Pkts/s', 'Fwd Header Len'
]

COLS_TO_DROP = [
    'Flow ID',
    'Source IP',
    'Source Port',
    'Destination IP',
    'Protocol',
    'Timestamp',
    'Flow Duration',
]

COLS = [
   'Flow ID', 'Source IP', 'Source Port', 'Destination IP',
   'Destination Port', 'Protocol', 'Timestamp', 'Flow Duration',
   'Total Fwd Packets', 'Total Backward Packets',
   'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
   'Fwd Packet Length Max', 'Fwd Packet Length Min',
   'Fwd Packet Length Mean', 'Fwd Packet Length Std',
   'Bwd Packet Length Max', 'Bwd Packet Length Min',
   'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
   'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
   'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
   'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
   'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
   'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
   'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
   'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
   'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
   'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
   'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
   'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
   'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
   'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
   'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
   'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
   'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
   'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
   'Idle Std', 'Idle Max', 'Idle Min', 'Label'
]


def clean_step(path_to_files: str, export_path: str, backup: bool = False) -> Tuple[pd.DataFrame, dict]:
    # Keep a trace of the cleaning step
    stats = defaultdict()
    stats["Dropped Columns"], stats["Dropped Negative Columns"], stats["Dropped NaN Columns"] = [], [], []
    stats["Negative Rows"], stats["NaN/INF Rows"] = 0, 0
    # 1- Merge all files
    print([path_to_files + '/' + f for f in os.listdir(path_to_files)])
    dfs = [pd.read_csv(path_to_files + '/' + f, compression='gzip') for f in os.listdir(path_to_files)]
    df = pd.concat(dfs)
    # df = pd.read_csv(path_to_files + '/' + 'USB-IDS-1-TRAINING.csv.gz')
    # Remove leading and trailing spaces from columns names
    df = df.rename(columns=dict(zip(df.columns, [col.strip() for col in df.columns])))
    total_rows = len(df)
    stats["Total Rows"] = str(total_rows)
    stats["Total Features"] = len(df.columns)

    # 2- Start data cleaning
    # 2.0- Remove pre-determined columns
    df = df.drop(COLS_TO_DROP, axis=1)
    stats["Dropped Columns"].extend(COLS_TO_DROP)

    # 2.1- Remove columns with unique values
    cols_uniq_vals = df.columns[df.nunique() <= 1].to_list()
    df = df.drop(cols_uniq_vals, axis=1)
    stats["Unique Columns"] = " ".join(cols_uniq_vals)

    # 2.2- Drop numerical columns with negative values
    num_cols = df.select_dtypes(exclude=["object", "category"]).columns.tolist()
    neg_cols = df[num_cols].columns[(df[num_cols] < 0).any()].tolist()
    stats["Negative Columns"] = " ".join(neg_cols)
    for col in neg_cols:
        neg_rows = (df[col] < 0).sum()
        if neg_rows >= 0.1 * len(df[col]):
            df = df.drop(col, axis=1)
            stats["Dropped Negative Columns"].append(col)
            stats["Dropped Columns"].append(col)
        else:
            stats["Negative Rows"] += neg_rows
            df.query(f'`{col}` >= 0', inplace=True)

    # Remove dropped columns from the numerical columns list
    if stats["Dropped Negative Columns"]:
        num_cols = list(set(num_cols) - set(stats["Dropped Negative Columns"]))

    # 2.3- Drop columns with NaN or INF values
    # Transforming all invalid data in numerical columns to NaN
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

    # 3- Converting labels to binary values
    df['Label'] = df['Label'].apply(lambda x: 1 if x == 'BENIGN' else 0)

    if backup:
        df.to_csv(
            f"{export_path}/{utils.folder_struct['clean_step']}/usb-ids_clean.csv",
            sep=',', encoding='utf-8', index=False
        )

    deleted_rows = stats["NaN/INF Rows"] + stats["Negative Rows"]
    stats["Ratio"] = f"{(deleted_rows / total_rows):1.4f}" if deleted_rows > 0 else "0.0"
    stats["Final Features"] = str(len(df.columns))
    stats["Final Total Rows"] = str(len(df))
    for key, val in stats.items():
        if type(val) == list:
            stats[key] = " ".join(val)
        elif type(val) != str:
            stats[key] = str(val)

    return df, stats


def normalize_step(df: pd.DataFrame, cols: list, base_path: str, fname: str, backup: bool = False):
    print(f'Processing {len(cols)} features for {fname}')
    # Preprocessing inspired by https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00426-w
    # Split numerical and non-numerical columns
    num_cols = df[cols].select_dtypes(exclude=["object", "category"]).columns.tolist()
    cat_cols = df[cols].select_dtypes(include=["category", "object"]).columns.tolist()
    # Optinally handle categorical values
    if cat_cols:
        print("Converting categorical columns " + ", ".join(cat_cols))
        perm = np.random.permutation(len(df))
        X = df.iloc[perm].reset_index(drop=True)
        y_prime = df['Label'].iloc[perm].reset_index(drop=True)
        enc = ce.CatBoostEncoder(verbose=1, cols=cat_cols)
        df = enc.fit_transform(X, y_prime)
    # Keep labels aside
    y = df['Label'].to_numpy()
    # Keep only a subset of the features
    df = df[cols]
    # Normalize numerical data
    scaler = MinMaxScaler()
    # Select numerical columns with values in the range (0, 1)
    # This way we avoid normalizing values that are already between 0 and 1.
    to_scale = df[num_cols][(df[num_cols] < 0.0).any(axis=1) & (df[num_cols] > 1.0).any(axis=1)].columns
    print(f"Scaling {len(to_scale)} columns: " + ", ".join(num_cols))
    df[to_scale] = scaler.fit_transform(df[to_scale].values.astype(np.float64))
    # Merge normalized dataframe with labels
    X = np.concatenate(
        (df.values, y.reshape(-1, 1)),
        axis=1
    )
    if backup:
        normalized_fname = f'{base_path}/{utils.folder_struct["normalize_step"]}/{fname}.csv'
        df.to_csv(
            normalized_fname,
            sep=',', encoding='utf-8', index=False
        )
        print(f'Saved {normalized_fname}')
        del df
    compressed_fname = f'{base_path}/{utils.folder_struct["minify_step"]}/{fname}.npz'
    np.savez(compressed_fname, usbids=X.astype(np.float64))
    print(f'Saved {compressed_fname}')


if __name__ == '__main__':
    # Assumes `path` points to the location of the original CSV files.
    # `path` must only contain CSV files and not other file types such as folders. 
    path, export_path, backup, _ = utils.parse_args()
    # 0 - Prepare folder structure
    utils.prepare(export_path)
    path_to_clean = f"{export_path}/{utils.folder_struct['clean_step']}/usb-ids_clean.csv"
    if os.path.isfile(path_to_clean):
        print("Clean file exists. Skipping cleaning step.")
        df = pd.read_csv(path_to_clean)
    else:
        # 1 - Clean the data (remove invalid rows and columns)
        df, clean_stats = clean_step(path, export_path, backup)
        # Save info about cleaning step
        utils.save_stats(export_path + '/usb_ids_info.csv', clean_stats)

    cols = df.columns.to_list()
    # 2 - Normalize numerical values and treat categorical values
    to_process = [
        (list(set(cols) - {'Label'}), 'feature_group_5'),
        # (list(set(cols) - {'Destination Port', 'Label'}), 'feature_group_5A'),
    ]
    df['Destination Port'] = df['Destination Port'].astype('category')
    for features, fname in to_process:
        normalize_step(df, features, export_path, fname, backup)
