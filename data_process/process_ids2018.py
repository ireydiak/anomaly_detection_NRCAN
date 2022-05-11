import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
import warnings
import utils
import os

warnings.filterwarnings('ignore')

NORMAL_LABEL = 0
ANORMAL_LABEL = 1

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
    'Bwd Byts/b Avg',
    'Bwd Pkts/b Avg',
    'Bwd Blk Rate Avg',
    'Bwd PSH Flags',
    'Bwd URG Flags',
    'Init Bwd Win Byts',
    'Dst IP',
    'Flow ID',
    'Src IP',
    'Src Port',
    'Flow Duration',
    'Protocol',
    'Timestamp',
    'Fwd Byts/b Avg',
    'Fwd Pkts/b Avg',
    'Fwd Blk Rate Avg',
    'Init Fwd Win Byts'
]

COLS = [
    'Dst Port',
    'Tot Fwd Pkts',
    'Tot Bwd Pkts',
    'TotLen Fwd Pkts',
    'TotLen Bwd Pkts',
    'Fwd Pkt Len Max',
    'Fwd Pkt Len Min',
    'Fwd Pkt Len Mean',
    'Fwd Pkt Len Std',
    'Bwd Pkt Len Max',
    'Bwd Pkt Len Min',
    'Bwd Pkt Len Mean',
    'Bwd Pkt Len Std',
    'Flow Byts/s',
    'Flow Pkts/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Flow IAT Max',
    'Flow IAT Min',
    'Fwd IAT Tot',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Fwd IAT Max',
    'Fwd IAT Min',
    'Bwd IAT Tot',
    'Bwd IAT Mean',
    'Bwd IAT Std',
    'Bwd IAT Max',
    'Bwd IAT Min',
    'Fwd PSH Flags',
    'Fwd URG Flags',
    'Fwd Header Len',
    'Bwd Header Len',
    'Fwd Pkts/s',
    'Bwd Pkts/s',
    'Pkt Len Min',
    'Pkt Len Max',
    'Pkt Len Mean',
    'Pkt Len Std',
    'Pkt Len Var',
    'FIN Flag Cnt',
    'SYN Flag Cnt',
    'RST Flag Cnt',
    'PSH Flag Cnt',
    'ACK Flag Cnt',
    'URG Flag Cnt',
    'CWE Flag Count',
    'ECE Flag Cnt',
    'Down/Up Ratio',
    'Pkt Size Avg',
    'Fwd Seg Size Avg',
    'Bwd Seg Size Avg',
    'Subflow Fwd Pkts',
    'Subflow Fwd Byts',
    'Subflow Bwd Pkts',
    'Subflow Bwd Byts',
    'Fwd Act Data Pkts',
    'Fwd Seg Size Min',
    'Active Mean',
    'Active Std',
    'Active Max',
    'Active Min',
    'Idle Mean',
    'Idle Std',
    'Idle Max',
    'Idle Min'
]


def clean_step(path_to_files: str, export_path: str) -> pd.DataFrame:
    total_rows = deleted_rows = 0
    total_features = 83
    chunks = []

    for f in os.listdir(path_to_files):
        print(f"Cleaning file {f}")
        chunk = pd.read_csv(f"{path_to_files}/{f}")
        total_rows += len(chunk)
        # Drop target columns if they exist
        chunk.drop(columns=COLS_TO_DROP, errors='ignore', inplace=True)

        # Transforming all non numeric values to NaN
        chunk[COLS] = chunk[COLS].apply(pd.to_numeric, errors='coerce')

        # Replacing INF values with NaN
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Filtering NaN values
        before_drop = len(chunk)
        chunk.dropna(inplace=True)
        deleted_rows += (before_drop - len(chunk))

        # Filtering negative values
        before_drop = len(chunk)
        chunk.query('`Fwd Header Len` >= 0', inplace=True)
        chunk.query('`Flow IAT Mean` >= 0', inplace=True)
        deleted_rows += (before_drop - len(chunk))

        # Converting labels to binary values
        chunk['Label_cat'] = chunk['Label']
        chunk['Label'] = chunk['Label'].apply(lambda x: NORMAL_LABEL if x == 'Benign' else ANORMAL_LABEL)

        # Adding chunk to chunks
        chunks.append(chunk)
        
        # backup
        # chunk.to_csv(
        #     f"{export_path}/{utils.folder_struct['clean_step']}/{f}",
        #     sep=',', encoding='utf-8', index=False
        # )


    stats = {
        "Total Rows": str(total_rows),
        "Total Features": "83",
        "Dropped Rows": str(deleted_rows),
        "Rows after clean": str(total_rows - deleted_rows),
        "Ratio": f"{(deleted_rows / total_rows):1.4f}",
        "Features after clean": str(len(chunk.columns))
    }
    return pd.concat(chunks), stats


def normalize_step(df: pd.DataFrame, cols: list, base_path: str, fname: str):
    print(f'Processing {len(cols)} features for {fname}')
    # Preprocessing inspired by https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00426-w
    # Split numerical and non-numerical columns
    num_cols = df[cols].select_dtypes(exclude=["object", "category"]).columns.tolist()
    cat_cols = df[cols].select_dtypes(include=["category", "object"]).columns.tolist()
    # Optinally handle categorical values
    if cat_cols:
        perm = np.random.permutation(len(df))
        X = df.iloc[perm].reset_index(drop=True)
        y_prime = df['Label'].iloc[perm].reset_index(drop=True)
        enc = ce.CatBoostEncoder(verbose=1, cols=cat_cols)
        df = enc.fit_transform(X, y_prime)
    # Keep labels aside
    y = df['Label'].to_numpy()
    source_label = df['Label_cat'].to_numpy()
    # Keep only a subset of the features
    df = df[cols]
    # Normalize numerical data
    scaler = MinMaxScaler()
    # Select numerical columns with values in the range (0, 1)
    # This way we avoid normalizing values that are already between 0 and 1.
    to_scale = df[num_cols][(df[num_cols] < 0.0).any(axis=1) & (df[num_cols] > 1.0).any(axis=1)].columns
    print(f'Scaling {len(to_scale)} columns')
    df[to_scale] = scaler.fit_transform(df[to_scale].values.astype(np.float))
    # Merge normalized dataframe with labels
    X = np.concatenate(
        (df.values, y.reshape(-1, 1)),
        axis=1
    )
    # df.to_csv(
    #     f'{base_path}/{utils.folder_struct["normalize_step"]}/{fname}.csv',
    #     sep=',', encoding='utf-8', index=False
    # )
    # print(f'Saved {base_path}/{utils.folder_struct["normalize_step"]}/{fname}.csv')
    del df

    np.savez_compressed(f'{base_path}/{utils.folder_struct["minify_step"]}/{fname}.npz', ids2018=X, label=source_label)
    # np.savez(f'{base_path}/{utils.folder_struct["minify_step"]}/{fname}_label.npz', ids2018=)
    print(f'Saved {base_path}/{fname}.npz')


if __name__ == '__main__':
    # Assumes `path` points to the location of the original CSV files.
    # `path` must only contain CSV files and not other file types such as folders.
    path, export_path, backup, _ = utils.parse_args()
    # 0 - Prepare folder structure
    utils.prepare(export_path)
    path_to_clean = f"{export_path}/{utils.folder_struct['clean_step']}/cicids2018_clean.csv"
    if os.path.isfile(path_to_clean):
        print("Clean file exists. Skipping cleaning step.")
        df = pd.read_csv(path_to_clean)
    else:
        # 1 - Clean the data (remove invalid rows and columns)
        df, clean_stats = clean_step(path, export_path)
        # Save info about cleaning step
        utils.save_stats(export_path + '/cicids2018_info.csv', clean_stats)
    
    # 2 - Normalize numerical values and treat categorical values
    to_process = [
        (list(set(COLS) - set(COLS_TO_DROP) - {'Label', 'Label_cat'}), 'feature_group_5'),
        # (["Dst Port", *rank_7_otf_7], 'feature_group_4'),
        # (["Dst Port", *rank_6_otf_7], 'feature_group_3'),
        # (["Dst Port", *rank_5_otf_7], 'feature_group_2'),
        # (["Dst Port", *rank_4_otf_7], 'feature_group_1'),
        # (list(set(COLS) - set(COLS_TO_DROP) - {'Dst Port', 'Label'}), 'feature_group_5A'),
        # (rank_7_otf_7, 'feature_group_4A'),
        # (rank_6_otf_7, 'feature_group_3A'),
        # (rank_5_otf_7, 'feature_group_2A'),
        # (rank_4_otf_7, 'feature_group_1A'),
    ]
    df['Label_cat'] = df['Label_cat'].astype('category')
    df['Dst Port'] = df['Dst Port'].astype('category')
    for features, fname in to_process:
        normalize_step(df, features, export_path, fname)
