import os

import pandas as pd
import numpy as np
import argparse
import utils

folder_struct = {
    'clean_step': '1_clean',
    'minify_step': '2_minified'
}

os.chdir(os.path.dirname(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output-directory', type=str, default='../data/')

NORMAL_LABEL = 0
ANORMAL_LABEL = 1


def export_stats(output_dir: str, stats: dict):
    with open(f'{output_dir}/kdd10percent_infos.csv', 'w') as f:
        f.write(','.join(stats.keys()) + '\n')
        f.write(','.join([str(val) for val in stats.values()]))


def df_stats(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['category', 'object']).columns
    return {
        'n_instances': df.shape[0],
        'in_features': df.shape[1],
        'Numerical Columns': len(num_cols),
        'Categorical Columns': len(cat_cols)
    }


def import_data():
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld"
    url_data = f"{base_url}/kddcup.data_10_percent.gz"
    url_info = f"{base_url}/kddcup.names"
    df_info = pd.read_csv(url_info, sep=":", skiprows=1, index_col=False, names=["colname", "type"])
    colnames = df_info.colname.values
    coltypes = np.where(df_info["type"].str.contains("continuous"), "float", "str")
    colnames = np.append(colnames, ["label"])
    coltypes = np.append(coltypes, ["str"])
    return pd.read_csv(url_data, names=colnames, index_col=False, dtype=dict(zip(colnames, coltypes)))


def preprocess(df: pd.DataFrame):
    # Dropping columns with unique values
    cols_uniq_vals = df.columns[df.nunique() <= 1]
    df = df.drop(cols_uniq_vals, axis=1)

    # One-hot encode the seven categorical attributes (except labels)
    # Assumes dtypes were previously assigned
    one_hot = pd.get_dummies(df.iloc[:, :-1])

    # Extract and simplify labels (normal data is 0, attacks are labelled as 1)
    y = np.where(df.label == "normal.", ANORMAL_LABEL, NORMAL_LABEL)

    # We know the anomaly ration should be around 20%
    assert np.isclose((y == ANORMAL_LABEL).sum() / len(y), .20, rtol=1.)

    X = np.concatenate(
        (one_hot.values, y.reshape(-1, 1)),
        axis=1
    )
    assert np.isnan(X).sum() == 0, "found nan values, aborting"

    return X, df, cols_uniq_vals.to_list()


def main():
    _, output_dir, _ = utils.parse_args()
    df_0 = import_data()
    stats_0 = df_stats(df_0)

    X, df_1, cols_dropped = preprocess(df_0)
    stats_0['Normal Instances'] = (df_1['label'] == NORMAL_LABEL).sum()
    stats_0['Anormal Instances'] = (df_1['label'] == ANORMAL_LABEL).sum()
    stats_0['Anomaly Ratio'] = stats_0['Anormal Instances'] / len(df_1)
    stats_1 = {'Final n_instances': len(X), 'Final n_features': X.shape[1] - 1}
    stats_1['N Dropped Columns'] = len(cols_dropped)
    stats_1['Dropped Columns'] = [' '.join(cols_dropped)]
    stats_1['NaN values'] = int(np.isnan(X).sum())

    # prepare the output directory
    utils.prepare(output_dir)

    export_stats(output_dir, dict(**stats_0, **stats_1))

    path = '{}/{}/{}'.format(output_dir, folder_struct["minify_step"], "kdd10percent")
    np.save(path, X.astype(np.float64))


if __name__ == '__main__':
    main()
