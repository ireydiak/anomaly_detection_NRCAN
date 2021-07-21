import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset-path', type=str)
parser.add_argument('-o', '--output-directory', help='Must be an empty directory', type=str)

folder_struct = {
    'clean_step': '1_clean',
    'normalize_step': '2_normalized',
    'minify_step': '3_minified'
}

NORMAL_LABEL = 1
ANORMAL_LABEL = 0

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
    return pd.read_csv(path, names=colnames, index_col=False, dtype=dict(zip(colnames, coltypes)))


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
    
    df_0 = import_data(args.dataset_path)
    stats_0 = df_stats(df_0)

    X, df_1, n_cols_dropped = preprocess(df_0)
    
    stats_0['Normal Instances'] = (df_1['label'] == NORMAL_LABEL).sum()
    stats_0['Anormal Instances'] = (df_1['label'] == ANORMAL_LABEL).sum()
    stats_1 = {'N Prime': len(X), 'D Prime': X.shape[1] - 1}
    stats_1['Dropped Columns'] = n_cols_dropped
    export_stats(args.output_directory, stats_0, stats_1)

    df_1.to_csv(
        '{}/{}/{}.csv'.format(args.output_directory, folder_struct["normalize_step"], "NSL-KDD_normalized"),
        sep=',', encoding='utf-8', index=False
    )
    np.savez(
        '{}/{}/{}.npz'.format(args.output_directory, folder_struct["minify_step"], "NSL-KDD_minified", 
        kdd=X.astype(np.float64))
    )
