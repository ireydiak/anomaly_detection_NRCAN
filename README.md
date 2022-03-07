## TODO
- Combine all `DataManager` classes in a single file.
- Make the `Trainer` classes extend the `BaseTrainer` class.
- Combine the models into fewer files (e.g. reconstruction-based models could be stored in `reconstruction.py`). 
Repeat the same process for `Trainer` classes.
- Simplify the main function. Each model should implement a `resolve` method thats take a dataset name as input. It
sets the network structure and different parameters based on the given dataset. See `model.NeuTraAD._resolve_params`
for reference. The command line arguments would reflect these changes (meaning we should have less arguments).
- Replace the `process_*` files within `data_process` by jupyter notebooks. The goal is to "see" every step
of the preprocessing pipeline (the file `process_ids2018.py` is a good starting point).
- Ideally, all experiments should have a jupyter notebook (found under `notebooks/experiments`)


# Deep unsupervised anomaly detection algorithms
This repository collects different unsupervised machine learning algorithms to detect anomalies.
## Implemented models
We have implemented the following models
- [x] [AutoEncoder]()
- [x] [DAGMM](https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf)
- [x] [SOM-DAGMM](https://arxiv.org/pdf/2008.12686.pdf)
- [x] [MemAE](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.pdf)
- [x] [DUAD]()
- [x] [NeuTraLAD]()
- 

## Dependencies
We recommend building an environment using [conda](https://docs.conda.io/en/latest/) as most of the dependencies come be preinstalled.
- minisom==2.2.9
- numpy==1.20.2
- pandas==1.2.4
- python=3.8.10
- pytorch=1.8.1
- scipy==1.6.2
- scikit-learn==0.24.2
- tqdm=4.59.0
- torchvision=0.9.1

## Installation
Assumes latest version of Anaconda was installed.
```
$ conda create -y --[ENV_NAME] py37 python=3.7
$ conda install --force-reinstall -y -q --name [ENV_NAME] -c conda-forge --file requirements.txt
$ conda activate [ENV_NAME]
```
Replace `[ENV_NAME]` with the name of your environment.

## Usage
```
$ python main.py -m [model] -d [/path/to/dataset] --dataset [dataset_name] -e [num_epochs]
```

Our model contains the following parameters:
- m: The selected machine learning model (**required**)
- d: The path to the dataset (**required**)
- dataset: The name selected dataset. Choices are `kdd`, `ids2018`, `nslkdd` (**required**).
- e: The number of training epochs
- batch-size: The size of a training batch
- optimizer: The optimizer algorithm used during training
- validation: Percentage of training data to use for validation
- lr: The learning rate  
- p-threshold: The percentile threshold for the energy
- lambda-energy: The DAGMM hyperparameter lambda_1
- lambda-p: The DAGMM hyperparameter lambda_2
- pct: The percentage of the original data to keep (useful for large datasets)
- rho: The anomaly ratio within the training set

## Example
To train a DAGMM on the KDD 10 percent dataset with the default parameters describe in the original paper:
```
$ python main.py -m DAGMM -d [/path/to/dataset.npz] --dataset kdd
```
Replace `[/path/to/dataset.npz]` with the path to the dataset in a numpy-friendly format.

