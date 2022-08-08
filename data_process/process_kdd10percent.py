import os
from collections import defaultdict

import pandas as pd
import numpy as np
import argparse
import utils

os.chdir(os.path.dirname(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output-directory', type=str, default='../data/')

NORMAL_LABEL = 0
ANORMAL_LABEL = 1
NORMAL_CAT = "normal."


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
        'Categorical Columns': len(cat_cols),
    }


def import_data():
    base_url = "http://kdd.ics.uci.edu/databases/kddcup99"
    url_data = f"{base_url}/kddcup.data_10_percent.gz"
    url_info = f"{base_url}/kddcup.names"
    df_info = pd.read_csv(url_info, sep=":", skiprows=1, index_col=False, names=["colname", "type"])
    colnames = df_info.colname.values
    coltypes = np.where(df_info["type"].str.contains("continuous"), "float", "str")
    colnames = np.append(colnames, ["label"])
    coltypes = np.append(coltypes, ["str"])
    return pd.read_csv(url_data, names=colnames, index_col=False, dtype=dict(zip(colnames, coltypes)))


def uniq_step(df: pd.DataFrame):
    # Dropping columns with unique values
    cols_uniq_vals = df.columns[df.nunique() <= 1]
    print("Dropped {} columns: {} (unique values)".format(len(cols_uniq_vals), cols_uniq_vals))
    df = df.drop(cols_uniq_vals, axis=1)
    return df, cols_uniq_vals


def preprocess(df: pd.DataFrame, stats: dict):
    # Drop columns with unique values
    df, dropped_cols = uniq_step(df)
    stats["uniq_cols"] = list(dropped_cols)

    # keep labels aside to avoid one-hot encoding them
    labels = df.label
    # One-hot encode the seven categorical attributes (except labels)
    # Assumes dtypes were previously assigned
    df = pd.get_dummies(df.iloc[:, :-1])
    df["label"] = labels
    # Keep the full labels aside before "binarizing" them
    df["Category"] = labels
    # Convert labels to binary labels
    df.loc[df["label"] == NORMAL_CAT, "label"] = 1
    df.loc[df["label"] != 1, "label"] = 0

    # We know the anomaly ration should be around 20%
    assert np.isclose(
        (df["label"] == ANORMAL_LABEL).sum() / len(df), .20,
        rtol=1.
    )

    assert df.isna().any(1).sum() == 0, "found nan values, aborting"

    X = df.drop("Category", axis=1).to_numpy()

    return df, X, stats


def main():
    _, output_dir, _ = utils.parse_args()
    df = import_data()
    stats = df_stats(df)

    df, X, stats = preprocess(df, stats)
    stats["Final n_instances"] = len(df)
    stats["Final n_features"] = df.shape[1] - 2
    stats["Anomaly Ratio"] = (df.label == 1).sum() / len(df)

    export_stats(output_dir, stats)

    path = os.path.join(output_dir, "kdd10percent.csv")
    df.to_csv(path)
    np.save(os.path.join(output_dir, "kdd10percent"), X.astype(np.float64))


if __name__ == '__main__':
    main()
