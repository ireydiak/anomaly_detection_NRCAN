conda activate nrcan &&
cd .. &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/alad.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/autoencoder.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/dagmm.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/deepsvdd.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/drocc.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/dsebm-e.yaml --save_dir experiments/training/LitDSEBM-e --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/dsebm-r.yaml --save_dir experiments/training/LitDSEBM-r --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/duad.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/goad.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/memae.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/_trainer.yaml --config models/IDS2018/neutralad.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/lof.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2018/_data.yaml --config models/IDS2018/oc-svm.yaml --save_dir experiments/training --n_runs 5