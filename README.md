# Anomaly Detection in Network Traffic
This repo collects different unsupervised machine learning methods to detect anomalies.
We are interested mostly in anomalies within Power Grid networks but the implemented methods can also be applied in different areas such as healthcare.  

## Implemented models
We have implemented the following models
- [x] [DAGMM](https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf): An autoencoder trained in an end-to-end fashion with a Gaussian Mixture Model to minimize the energy of the sample space. Based on the model proposed by Zong et al.
- [ ] MLAD: An unpublished model develop by our colleague Jianhai.
- [ ] [SOM-DAGMM](https://arxiv.org/pdf/2008.12686.pdf): Concatenates features of a DAGMM (reconstruction error and latent space of the AutoEncoder) with the features taken from a 2x2 Self-Organizing Map. Based on the model proposed by Chen et al.

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
$ python main.py -m [model] -d [/path/to/dataset] --dataset [dataset_name]
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
