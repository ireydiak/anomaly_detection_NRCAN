import gzip
import os
import pickle

import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler
import utils

folder_struct = {
    'clean_step': '1_clean',
    'normalize_step': '2_normalized',
    'minify_step': '3_minified'
}

os.chdir(os.path.dirname(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output-directory', type=str, default='../data/')

NORMAL_LABEL = 0
ANORMAL_LABEL = 1


def process_step(path_to_file, seq_length=50, pad=True, bloc_size=660):
    data = []
    labels = []
    seq = []
    counter = 0
    label = 0
    with open(path_to_file, 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                if 0 < len(seq) <= seq_length:
                    counter += 1
                    # padding with zero to get sequence of equal size
                    if pad:
                        offset = seq_length - len(seq)
                        offsetlist = [[0] * len(seq[0]) for _ in range(offset)]
                        seq.extend(offsetlist)
                    data.append(seq)
                    labels.append(label)
                    if counter % bloc_size == 0:
                        label += 1
                seq = []
            else:
                seq.append(line.split())

    return data, labels


if __name__ == '__main__':
    path_to_files, export_path, backup, normalize = utils.parse_args()
    DATASET_NAME = 'sad'

    # 0 - Prepare folder structure
    utils.prepare(export_path)
    dataset = dict()
    # 0 - Processing of train and test files
    print('preprocessing')
    for f in os.listdir(path_to_files):
        file_path = path_to_files + '/' + f
        bloc_size = 660 if 'train' in f.lower() else 220
        X, y = process_step(file_path, bloc_size=bloc_size)
        data_type = 'train' if 'train' in f.lower() else 'test'
        dataset[data_type] = dict(X=X, y=y)

    # save data to file (.pklz)
    print('saving data\n========')
    with gzip.open(f'{export_path}/{DATASET_NAME}.pklz', 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('completed')
