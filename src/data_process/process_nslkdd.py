import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler

from src.utils import check_dir

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset-path', type=str)
parser.add_argument('-o', '--output-directory', help='Must be an empty directory', type=str)

TRAIN_FILENAME = 'KDDTrain+.txt'
TEST_FILENAME = 'KDDTest+.txt'

folder_struct = {
    'clean_step': '1_clean',
    'normalize_step': '2_normalized',
    'minify_step': '3_minified'
}

NORMAL_LABEL = 0
ANORMAL_LABEL = 1

def export_stats(output_dir: str, before_stats: dict, after_stats: dict):
    after_stats = {f'{key} Prime': val for key, val in after_stats.items()}
    stats = dict(**before_stats, **after_stats)
    with open(f'{output_dir}/nsl-kdd_infos.csv', 'w') as f:
        f.write(','.join(stats.keys()) + '\n')
        f.write(','.join([str(val) for val in stats.values()]))


def df_stats(df: pd.DataFrame):
    stats = df.dtypes.value_counts()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['category', 'object']).columns
    return {
        'D': df.shape[1],
        'N': df.shape[0],
        'Numerical Columns': len(num_cols),
        'Categorical Columns': len(cat_cols)
    }


def import_data(path: str):
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld"
    url_info = f"{base_url}/kddcup.names"
    df_info = pd.read_csv(url_info, sep=":", skiprows=1, index_col=False, names=["colname", "type"])
    colnames = df_info.colname.values
    coltypes = np.where(df_info["type"].str.contains("continuous"), "float", "str")
    colnames = np.append(colnames, ["label"])
    coltypes = np.append(coltypes, ["str"])
    df_train = pd.read_csv(path + '/' + TRAIN_FILENAME, names=colnames, index_col=False, dtype=dict(zip(colnames, coltypes)))
    df_test = pd.read_csv(path + '/' + TEST_FILENAME, names=colnames, index_col=False, dtype=dict(zip(colnames, coltypes)))
    # UBN added an extra `difficulty_level` columns which we ignore here
    return pd.concat([df_train[:, :-1], df_test[:, :-1]], ignore_index=True, sort=False)

def preprocess(df: pd.DataFrame):
    # Dropping columns with unique values
    cols_uniq_vals = df.columns[df.nunique() <= 1]
    df = df.drop(cols_uniq_vals, axis=1)

    # One-hot encode the seven categorical attributes (except labels)
    # Assumes dtypes were previously assigned
    one_hot = pd.get_dummies(df.iloc[:, :-1])

    # min-max scaling
    scaler = MinMaxScaler()
    cols = one_hot.select_dtypes(["float", "int"]).columns
    one_hot[cols] = scaler.fit_transform(one_hot[cols].values.astype(np.float64))

    # Extract and simplify labels (normal data is 1, attacks are labelled as 0)
    y = np.where(df.label == "normal", NORMAL_LABEL, ANORMAL_LABEL)
    df['label'] = y

    X = np.concatenate(
        (one_hot.values, y.reshape(-1, 1)),
        axis=1
    )

    return X, df, len(cols_uniq_vals)


if __name__ == '__main__':
    args = parser.parse_args()
    
    dataset_path = args.dataset_path[:-1] if args.dataset_path.endswith('/') else args.dataset_path
    df_0 = import_data(dataset_path)
    stats_0 = df_stats(df_0)

    X, df_1, n_cols_dropped = preprocess(df_0)
    
    stats_0['Normal Instances'] = (df_1['label'] == NORMAL_LABEL).sum()
    stats_0['Anormal Instances'] = (df_1['label'] == ANORMAL_LABEL).sum()
    stats_0['Anomaly Ratio'] = stats_0['Anormal Instances'] / len(df_1)
    stats_1 = {'N Prime': len(X), 'D Prime': X.shape[1] - 1}
    stats_1['Dropped Columns'] = n_cols_dropped
    export_stats(args.output_directory, stats_0, stats_1)

    path = '{}/{}/{}.csv'.format(args.output_directory, folder_struct["normalize_step"], "NSL-KDD_normalized")
    # check if the directory exists or create it
    check_dir(path)
    df_1.to_csv(
        path,
        sep=',', encoding='utf-8', index=False
    )

    path = '{}/{}/{}.npz'.format(args.output_directory, folder_struct["minify_step"], "NSL-KDD_minified")
    # check if the directory exists or create it
    check_dir(path)
    np.savez(
        path,
        kdd=X.astype(np.float64)
    )
