cd .. &&
eval "$(conda shell.bash hook)" &&
conda activate nrcan &&
python -m pyad.lightning_cli --config models/KDD10/_data.yaml --config models/KDD10/_trainer.yaml --config models/KDD10/alad.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/KDD10/_data.yaml --config models/KDD10/_trainer.yaml --config models/KDD10/autoencoder.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/KDD10/_data.yaml --config models/KDD10/_trainer.yaml --config models/KDD10/dagmm.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/KDD10/_data.yaml --config models/KDD10/_trainer.yaml --config models/KDD10/deepsvdd.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/KDD10/_data.yaml --config models/KDD10/_trainer.yaml --config models/KDD10/drocc.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/KDD10/_data.yaml --config models/KDD10/_trainer.yaml --config models/KDD10/dsebm-e.yaml --save_dir experiments/training/LitDSEBM-e --n_runs 5 &&
python -m pyad.lightning_cli --config models/KDD10/_data.yaml --config models/KDD10/_trainer.yaml --config models/KDD10/dsebm-r.yaml --save_dir experiments/training/LitDSEBM-r --n_runs 5 &&
python -m pyad.lightning_cli --config models/KDD10/_data.yaml --config models/KDD10/_trainer.yaml --config models/KDD10/duad.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/KDD10/_data.yaml --config models/KDD10/_trainer.yaml --config models/KDD10/goad.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/KDD10/_data.yaml --config models/KDD10/_trainer.yaml --config models/KDD10/memae.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/KDD10/_data.yaml --config models/KDD10/_trainer.yaml --config models/KDD10/neutralad.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/KDD10/_data.yaml --config models/KDD10/lof.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/KDD10/_data.yaml --config models/KDD10/oc-svm.yaml --save_dir experiments/training --n_runs 5