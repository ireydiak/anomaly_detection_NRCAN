import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output-file', type=str, default='../data/kdd10percent.npz')


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

    # min-max scaling
    scaler = MinMaxScaler()
    cols = one_hot.select_dtypes(["float", "int"]).columns
    one_hot[cols] = scaler.fit_transform(one_hot[cols].values.astype(np.float64))

    # Extract and simplify labels (normal data is 1, attacks are labelled as 0)
    y = np.where(df.label == "normal.", 1, 0)

    X = np.concatenate(
        (one_hot.values, y.reshape(-1, 1)),
        axis=1
    )

    return X


if __name__ == '__main__':
    args = parser.parse_args()
    X = preprocess(import_data())
    np.savez(args.output_file, kdd=X.astype(np.float64))
