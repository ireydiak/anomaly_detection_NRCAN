cd .. &&
eval "$(conda shell.bash hook)" &&
conda activate nrcan &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/alad.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/autoencoder.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/dagmm.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/deepsvdd.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/drocc.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/dsebm-e.yaml --save_dir experiments/training/LitDSEBM-e --n_runs 5 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/dsebm-r.yaml --save_dir experiments/training/LitDSEBM-r --n_runs 5 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/duad.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/goad.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/memae.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/neutralad.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/lof.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/oc-svm.yaml --save_dir experiments/training --n_runs 5