# Deep unsupervised anomaly detection algorithms
This repository collects different unsupervised machine learning algorithms to detect anomalies.
## Implemented models
We have implemented the following models. Our implementations of ALAD, DeepSVDD, 
DROCC and MemAE closely follows the original implementations already available on GitHub.
- [x] [AutoEncoder]()
- [x] [ALAD](https://arxiv.org/abs/1812.02288)
- [x] [DAGMM](https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf)
- [x] [DeepSVDD](http://proceedings.mlr.press/v80/ruff18a.html)
- [x] [DSEBM](https://arxiv.org/abs/1605.07717)
- [x] [DROCC](https://arxiv.org/abs/2002.12718)
- [x] [DUAD](https://openaccess.thecvf.com/content/WACV2021/papers/Li_Deep_Unsupervised_Anomaly_Detection_WACV_2021_paper.pdf)
- [x] [LOF](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
- [x] [MemAE](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.pdf)
- [x] [NeuTraLAD](https://arxiv.org/pdf/2103.16440.pdf)
- [x] [OC-SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)
- [x] [RecForest](https://epubs.siam.org/doi/pdf/10.1137/1.9781611976700.15)
- [x] [SOM-DAGMM](https://arxiv.org/pdf/2008.12686.pdf)

## Suggested parameters
#### DAGMM
| Dataset    | n_mixtures | latent_dim | lambda_1 | lambda_2 | reg_covar |
|------------|------------|------------|----------|----------|-----------|
| Arrhythmia | 2          | 1          | 0.005    | 0.1      | 1e-12     |
| Thyroid    | 4          | 1          | 0.005    | 0.1      | 1e-12     |
| Default    | 4          | 1          | 0.005    | 0.1      | 1e-12     |
 
#### AutoEncoder
| Dataset    | latent_dim | compression_factor | n_layers | act_fn |
|------------|------------|--------------------|----------|--------|
| Arrhythmia | 10         | 1                  | 2        | tanh   |
| Thyroid    | 1          | 2                  | 3        | tanh   |
| Default    | 1          | 0.005              | 4        | relu   |


## Dependencies
A complete dependency list is available in requirements.txt.
We list here the most important ones:
- torch@1.10.2 with CUDA 11.3
- numpy
- pandas
- scikit-learn

## Installation
Assumes latest version of Anaconda was installed.
```
$ conda create --name [ENV_NAME] python=3.8
$ conda activate [ENV_NAME]
$ pip install -r requirements.txt
```
Replace `[ENV_NAME]` with the name of your environment.

## Usage
From the root of the project.
```
$ python -m src.main 
-m [model_name]
-d [/path/to/dataset/file.{npz,mat}]
--dataset [dataset_name]
--batch-size [batch_size]
```

Our model contains the following parameters:
- `-m`: selected machine learning model (**required**)
- `-d`: path to the dataset (**required**)
- `--batch-size`: size of a training batch (**required**)
- `--dataset`: name of the selected dataset. Choices are `Arrhythmia`, `KDD10`, `IDS2018`, `NSLKDD`, `USBIDS`, `Thyroid` (**required**).
- `-e`: number of training epochs (default=200)
- `--n-runs`: number of time the experiment is repeated (default=1)
- `--lr`: learning rate used during optimization (default=1e-4)
- `--pct`: percentage of the original data to keep (useful for large datasets, default=1.)
- `rho`: anomaly ratio within the training set (default=0.)
- `--results-path`: path where the results are stored (default="../results")
- `--model-path`: path where models will be stored (default="../models")
- `--test-mode`: loads models from `--model_path` and tests them (default=False)
Please note that datasets must be stored in `.npz` or `.mat` files. Use the preprocessing scripts within `data_process`
to generate these files.

## Example
To train a DAGMM on the KDD 10 percent dataset with the default parameters described in the original paper:
```
$ python  -m src.main -m DAGMM -d [/path/to/dataset.npz] --dataset KDD10 --batch-size 1024 --results-path ./results/KDD10 --models-path ./models/KDD10
```
Replace `[/path/to/dataset.npz]` with the path to the dataset in a numpy-friendly format.

Optionally, a Jupyter notebook is made available in `experiments.ipynb`
