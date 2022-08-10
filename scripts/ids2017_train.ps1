conda activate nrcan &&
cd .. &&
python -m pyad.lightning_cli --config models/IDS2017/_data.yaml --config models/IDS2017/_trainer.yaml --config models/IDS2017/lof.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2017/_data.yaml --config models/IDS2017/_trainer.yaml --config models/IDS2017/oc-svm.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2017/_data.yaml --config models/IDS2017/_trainer.yaml --config models/IDS2017/alad.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2017/_data.yaml --config models/IDS2017/_trainer.yaml --config models/IDS2017/autoencoder.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2017/_data.yaml --config models/IDS2017/_trainer.yaml --config models/IDS2017/deepsvdd.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2017/_data.yaml --config models/IDS2017/_trainer.yaml --config models/IDS2017/drocc.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2017/_data.yaml --config models/IDS2017/_trainer.yaml --config models/IDS2017/dsebm-e.yaml --save_dir experiments/training/LitDSEBM-e --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2017/_data.yaml --config models/IDS2017/_trainer.yaml --config models/IDS2017/dsebm-r.yaml --save_dir experiments/training/LitDSEBM-r --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2017/_data.yaml --config models/IDS2017/_trainer.yaml --config models/IDS2017/goad.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2017/_data.yaml --config models/IDS2017/_trainer.yaml --config models/IDS2017/memae.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2017/_data.yaml --config models/IDS2017/_trainer.yaml --config models/IDS2017/neutralad.yaml --save_dir experiments/training --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2017/_data.yaml --config models/IDS2017/_trainer.yaml --config models/IDS2017/duad.yaml --save_dir --n_runs 5 &&
python -m pyad.lightning_cli --config models/IDS2017/_data.yaml --config models/IDS2017/_trainer.yaml --config models/IDS2017/dagmm.yaml --save_dir experiments/training --n_runs 5 &&
