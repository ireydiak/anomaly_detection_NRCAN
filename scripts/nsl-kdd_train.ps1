conda activate nrcan &&
cd .. &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/alad.yaml --save_dir experiments/training --n_runs 20 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/autoencoder.yaml --save_dir experiments/training --n_runs 20 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/dagmm.yaml --save_dir experiments/training --n_runs 20 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/deepsvdd.yaml --save_dir experiments/training --n_runs 20 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/drocc.yaml --save_dir experiments/training --n_runs 20 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/dsebm-e.yaml --save_dir experiments/training --n_runs 20 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/dsebm-r.yaml --save_dir experiments/training --n_runs 20 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/goad.yaml --save_dir experiments/training --n_runs 20 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/memae.yaml --save_dir experiments/training --n_runs 20 &&
python -m pyad.lightning_cli --config models/NSL-KDD/_data.yaml --config models/NSL-KDD/_trainer.yaml --config models/NSL-KDD/neutralad.yaml --save_dir experiments/training --n_runs 20