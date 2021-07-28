import os
import argparse
from typing import Tuple

folder_struct = {
    'clean_step': '1_clean',
    'normalize_step': '2_normalized',
    'minify_step': '3_minified'
}


def parse_args() -> Tuple[str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--path', type=str,
        default=None,
        help='Path to original CSV file or path to root directory containing CSV files'
    )
    parser.add_argument(
        '-o', '--export-path', type=str,
        help='Path to the output directory. Folders will be added to this directory.'
    )

    args = parser.parse_args()
    return args.path if not args.path.endswith('/') else args.path[:-1], \
           args.export_path if not args.export_path.endswith('/') else args.export_path[:-1]


def prepare(base_path: str):
    folders = ['{}/{}'.format(base_path, folder) for _, folder in folder_struct.items()]
    for f in folders:
        if os.path.exists(f):
            print(f'Directory {f} already exists. Skipping')
        else:
            os.mkdir(f)

def save_stats(path: str, *stats: dict):
    vals = {k: v for d in stats for k, v in d.items()}
    with open(path, 'w') as f:
        f.write(','.join(vals.keys()) + '\n')
        f.write(','.join([str(val) for val in vals.values()]))