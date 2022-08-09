conda activate anomaly_detection &&
cd .. &&
python -m pyad.lightning_cli --config models/Arrhythmia/_data.yaml --config models/Arrhythmia/_trainer.yaml --config models/Arrhythmia/alad.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Arrhythmia/_data.yaml --config models/Arrhythmia/_trainer.yaml --config models/Arrhythmia/autoencoder.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Arrhythmia/_data.yaml --config models/Arrhythmia/_trainer.yaml --config models/Arrhythmia/dagmm.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Arrhythmia/_data.yaml --config models/Arrhythmia/_trainer.yaml --config models/Arrhythmia/deepsvdd.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Arrhythmia/_data.yaml --config models/Arrhythmia/_trainer.yaml --config models/Arrhythmia/drocc.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Arrhythmia/_data.yaml --config models/Arrhythmia/_trainer.yaml --config models/Arrhythmia/dsebm-e.yaml --save_dir experiments/training/Arrhythmia/dsebm-e --n_runs 1 &&
python -m pyad.lightning_cli --config models/Arrhythmia/_data.yaml --config models/Arrhythmia/_trainer.yaml --config models/Arrhythmia/dsebm-r.yaml --save_dir experiments/training/Arrhythmia/dsebm-r --n_runs 1 &&
python -m pyad.lightning_cli --config models/Arrhythmia/_data.yaml --config models/Arrhythmia/_trainer.yaml --config models/Arrhythmia/duad.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Arrhythmia/_data.yaml --config models/Arrhythmia/_trainer.yaml --config models/Arrhythmia/goad.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Arrhythmia/_data.yaml --config models/Arrhythmia/_trainer.yaml --config models/Arrhythmia/memae.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Arrhythmia/_data.yaml --config models/Arrhythmia/_trainer.yaml --config models/Arrhythmia/neutralad.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Arrhythmia/_data.yaml --config models/Arrhythmia/lof.yaml --save_dir experiments/training --n_runs 1 &&
python -m pyad.lightning_cli --config models/Arrhythmia/_data.yaml --config models/Arrhythmia/oc-svm.yaml --save_dir experiments/training --n_runs 1