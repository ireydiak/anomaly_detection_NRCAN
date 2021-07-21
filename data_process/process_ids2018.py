import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
import warnings
import os
warnings.filterwarnings('ignore')

NORMAL_LABEL = 1
ANORMAL_LABEL = 0

parser = argparse.ArgumentParser(
    description='CICIDS2018 preprocessing script.Assumes that header row duplicates were removed and original files were concatenated. Use merge.bash to concatenate CSV files (works only on Linux)',
    usage='\npython3 main.py [path] [export-path]'
)

parser.add_argument('-d', '--path', type=str, help='Path to original CSV file')
parser.add_argument('-o', '--export-path', type=str, help='Path to the output directory. Folders will be added to this directory.')

folder_struct = {
    'clean_step': '1_clean',
    'normalize_step': '2_normalized',
    'minify_step': '3_minified'
}

NON_NUMERIC_COLS = ['Label', 'Timestamp']
COLS_TO_DROP = ['Protocol', 'Timestamp', 'Flow Duration', 'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Flow Duration']
NEG_COLS = ['Fwd Header Len', 'Flow Duration', 'Fwd IAT Min']
INF_COLS = ['Flow Byts/s', 'Flow Pkts/s']
UNIQ_COLS = ['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg',
             'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg']

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

TYPES = {
    'Dst Port': 'uint32',
    'Protocol': 'uint8',
    'Timestamp': 'object',
    'Flow Duration': 'int64',
    'Tot Fwd Pkts': 'uint32',
    'Tot Bwd Pkts': 'uint32',
    'TotLen Fwd Pkts': 'uint32',
    'TotLen Bwd Pkts': 'uint32',
    'Fwd Pkt Len Max': 'uint16',
    'Fwd Pkt Len Min': 'uint16',
    'Fwd Pkt Len Mean': 'float32',
    'Fwd Pkt Len Std': 'float32',
    'Bwd Pkt Len Max': 'uint16',
    'Bwd Pkt Len Min': 'uint16',
    'Bwd Pkt Len Mean': 'float32',
    'Bwd Pkt Len Std': 'float32',
    'Flow Byts/s': 'float64',
    'Flow Pkts/s': 'float64',
    'Flow IAT Mean': 'float32',
    'Flow IAT Std': 'float32',
    'Flow IAT Max': 'int64',
    'Flow IAT Min': 'int64',
    'Fwd IAT Tot': 'int64',
    'Fwd IAT Mean': 'float32',
    'Fwd IAT Std': 'float32',
    'Fwd IAT Max': 'int64',
    'Fwd IAT Min': 'int64',
    'Bwd IAT Tot': 'uint32',
    'Bwd IAT Mean': 'float32',
    'Bwd IAT Std': 'float32',
    'Bwd IAT Max': 'uint32',
    'Bwd IAT Min': 'int64',
    'Fwd PSH Flags': 'uint8',
    'Bwd PSH Flags': 'uint8',
    'Fwd URG Flags': 'uint8',
    'Bwd URG Flags': 'uint8',
    'Fwd Header Len': 'uint32',
    'Bwd Header Len': 'uint32',
    'Fwd Pkts/s': 'float32',
    'Bwd Pkts/s': 'float32',
    'Pkt Len Min': 'uint16',
    'Pkt Len Max': 'uint16',
    'Pkt Len Mean': 'float32',
    'Pkt Len Std': 'float32',
    'Pkt Len Var': 'float32',
    'FIN Flag Cnt': 'uint16',
    'SYN Flag Cnt': 'uint16',
    'RST Flag Cnt': 'uint16',
    'PSH Flag Cnt': 'uint16',
    'ACK Flag Cnt': 'uint16',
    'URG Flag Cnt': 'uint16',
    'CWE Flag Count': 'uint16',
    'ECE Flag Cnt': 'uint16',
    'Down/Up Ratio': 'uint16',
    'Pkt Size Avg': 'float32',
    'Fwd Seg Size Avg': 'float32',
    'Bwd Seg Size Avg': 'float32',
    'Fwd Byts/b Avg': 'uint8',
    'Fwd Pkts/b Avg': 'uint8',
    'Fwd Blk Rate Avg': 'uint8',
    'Bwd Byts/b Avg': 'uint8',
    'Bwd Pkts/b Avg': 'uint8',
    'Bwd Blk Rate Avg': 'uint8',
    'Subflow Fwd Pkts': 'uint32',
    'Subflow Fwd Byts': 'uint32',
    'Subflow Bwd Pkts': 'uint32',
    'Subflow Bwd Byts': 'uint32',
    'Init Fwd Win Byts': 'int32',
    'Init Bwd Win Byts': 'int32',
    'Fwd Act Data Pkts': 'uint32',
    'Fwd Seg Size Min': 'uint8',
    'Active Mean': 'float32',
    'Active Std': 'float32',
    'Active Max': 'uint32',
    'Active Min': 'uint32',
    'Idle Mean': 'float32',
    'Idle Std': 'float32',
    'Idle Max': 'uint64',
    'Idle Min': 'uint64',
    'Label': 'category'
}

def convert_dtype(x, dtype):
    fn = getattr(np, dtype)

    try:
        return fn(x)
    except:
        return np.nan

def cols_to_drop() -> list:
    return UNIQ_COLS + COLS_TO_DROP


def find_invalid_rows(df: pd.DataFrame) -> list:
    inf_nan_rows = df.index[np.isinf(df[INF_COLS].fillna(np.inf)).any(1)].tolist()
    neg_rows = df.index[(df[NEG_COLS] < 0).any(1)].tolist()
    nan_rows = df.index[df.isna().any(1)].tolist()

    return list(set(inf_nan_rows) | set(neg_rows) | set(nan_rows))


def process(df: pd.DataFrame):
    to_drop = cols_to_drop()
    rows = find_invalid_rows(df)

    return df.drop(rows).drop(to_drop, axis=1), len(rows)


def parse_args():
    args = parser.parse_args()
    return args.path, \
           args.export_path if not args.export_path.endswith('/') else args.export_path[0:-1]


def clean_step(path: str, export_path: str) -> pd.DataFrame:
    total_rows = dropped_rows = 0
    df_final = pd.DataFrame()
    for i, chunk in enumerate(pd.read_csv(path, sep=',', chunksize=10 ** 6, dtype=TYPES)):
        print(f'Processing chunk {i + 1}')
        df_modified, nrows = process(chunk)
        df_final = pd.concat([df_final, df_modified])
        total_rows += chunk.shape[0]
        print(f'Found {nrows} invalid rows')
        dropped_rows += nrows

    df_final['Label'] = df_final['Label'].apply(lambda x: NORMAL_LABEL if x == 'Benign' else ANORMAL_LABEL)
    print(f'Total rows before changes: {total_rows}')
    print(f'Total of affected rows: {dropped_rows}')
    print(f'Total rows after changes: {df_final.shape[0]}')
    print('Percentage of rows affected: %8.2f' % ((dropped_rows * 100) / total_rows))
    df_final.to_csv(
        f'{export_path}/{folder_struct["clean_step"]}/cicids2018_clean.csv',
        sep=',', encoding='utf-8', index=False
    )
    print(f'Finished writing dataframe to {fname}')
    return df_final


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
    # Keep only a subset of the features
    df = df[cols]
    # Normalize numerical data
    scaler = MinMaxScaler()
    # Select numerical columns with values in the range (0, 1)
    # This way we avoid normalizing values that are already between 0 and 1.
    to_scale = df[num_cols][(df[num_cols] < 0.0).any(axis=1) & (df[num_cols] > 1.0).any(axis=1)].columns
    print(f'Scaling {len(to_scale)} columns')
    df[to_scale] = scaler.fit_transform(df[to_scale].values.astype(np.float64))
    # Merge normalized dataframe with labels
    X = np.concatenate(
        (df.values, y.reshape(-1, 1)),
        axis=1
    )
    df.to_csv(
        f'{base_path}/{folder_struct["normalize_step"]}/{fname}.csv',
        sep=',', encoding='utf-8', index=False
    )
    print(f'Saved {base_path}/{folder_struct["normalize_step"]}/{fname}.csv')
    del df
    np.savez(f'{base_path}/{folder_struct["minify_step"]}/{fname}.npz', ids2018=X.astype(np.float64))
    print(f'Saved {base_path}/{fname}.npz')

def prepare(base_path: str):
    os.makedirs(['{}/{}'.format(base_path, folder) for _, folder in folder_struct.items()])

if __name__ == '__main__':
    path, export_path = parse_args()
    # 0 - Prepare folder structure
    prepare(export_path)
    # 1 - Clean the data (remove invalid rows and columns)
    df = clean_step(path, export_path)
    # 2 - Normalize numerical values and treat categorical values
    to_process = [
        (list(set(TYPES.keys()) - set(cols_to_drop()) - {'Dst Port', 'Label'}), 'feature_group_5'),
        (["Dst Port", *rank_7_otf_7], 'feature_group_4'),
        (["Dst Port", *rank_6_otf_7], 'feature_group_3'),
        (["Dst Port", *rank_5_otf_7], 'feature_group_2'),
        (["Dst Port", *rank_4_otf_7], 'feature_group_1'),
        (list(set(TYPES.keys()) - set(cols_to_drop()) - {'Dst Port', 'Label'}), 'feature_group_5A'),
        (rank_7_otf_7, 'feature_group_4A'),
        (rank_6_otf_7, 'feature_group_3A'),
        (rank_5_otf_7, 'feature_group_2A'),
        (rank_4_otf_7, 'feature_group_1A'),
    ]
    df['Dst Port'] = df['Dst Port'].astype('category')
    for features, fname in to_process:
        normalize_step(df, features, export_path, fname)
