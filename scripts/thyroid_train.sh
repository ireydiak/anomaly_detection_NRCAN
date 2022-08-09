cd .. &&
eval "$(conda shell.bash hook)" &&
conda activate nrcan &&
python -m pyad.lightning_cli --config models/Thyroid/_data.yaml --config models/Thyroid/_trainer.yaml --config models/Thyroid/alad.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Thyroid/_data.yaml --config models/Thyroid/_trainer.yaml --config models/Thyroid/autoencoder.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Thyroid/_data.yaml --config models/Thyroid/_trainer.yaml --config models/Thyroid/dagmm.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Thyroid/_data.yaml --config models/Thyroid/_trainer.yaml --config models/Thyroid/deepsvdd.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Thyroid/_data.yaml --config models/Thyroid/_trainer.yaml --config models/Thyroid/drocc.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Thyroid/_data.yaml --config models/Thyroid/_trainer.yaml --config models/Thyroid/dsebm-e.yaml --save_dir experiments/training/Thyroid/dsebm-e --n_runs 1 &&
python -m pyad.lightning_cli --config models/Thyroid/_data.yaml --config models/Thyroid/_trainer.yaml --config models/Thyroid/dsebm-r.yaml --save_dir experiments/training/Thyroid/dsebm-r --n_runs 1 &&
python -m pyad.lightning_cli --config models/Thyroid/_data.yaml --config models/Thyroid/_trainer.yaml --config models/Thyroid/duad.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Thyroid/_data.yaml --config models/Thyroid/_trainer.yaml --config models/Thyroid/goad.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Thyroid/_data.yaml --config models/Thyroid/_trainer.yaml --config models/Thyroid/memae.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Thyroid/_data.yaml --config models/Thyroid/_trainer.yaml --config models/Thyroid/neutralad.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Thyroid/_data.yaml --config models/Thyroid/lof.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Thyroid/_data.yaml --config models/Thyroid/oc-svm.yaml --save_dir experiments/training --n_runs 1