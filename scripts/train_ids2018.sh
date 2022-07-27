cd .. &&
eval "$(conda shell.bash hook)" &&
conda activate $AD_CONDA_ENV &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/alad.yaml --save_dir experiments/training --n_runs 20 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/autoencoder.yaml --save_dir experiments/training --n_runs 20 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/dagmm.yaml --save_dir experiments/training --n_runs 20 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/deepsvdd.yaml --save_dir experiments/training --n_runs 20 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/drocc.yaml --save_dir experiments/training --n_runs 20 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/dsebm-e.yaml --save_dir experiments/training/dsebm-e --n_runs 20 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/dsebm-r.yaml --save_dir experiments/training/dsebm-r --n_runs 20 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/goad.yaml --save_dir experiments/training --n_runs 20 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/memae.yaml --save_dir experiments/training --n_runs 20 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/neutralad.yaml --save_dir experiments/training --n_runs 20