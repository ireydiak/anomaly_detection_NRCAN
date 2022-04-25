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

## Parameters
Our codebase was made to maximize the number of parameters a user can set without modifying the code while minimizing the argparse footprint.

Global parameters are:
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

Model specific parameters are:
- AE 
  - `--ae-latent-dim`: latent dimension of the network
    - Default: 1
  - `--ae-n-layers`: number of layers of the network
    - Default: 4
  - `--ae-compression-factor`: compression factor for the network
    - Default: 2
  - `--ae-act-fn`: activation function of the network
    - Default: `relu`
- DAGMM
  - `--dagmm-latent-dim`: latent dimension of the AE subnetwork
    - Default: `1`
  - `--dagmm-n-mixtures`: number of mixtures for the GMM network
    - Default: `4`
  - `--dagmm-n-layers`: number of layers of the AE network
    - Default: `4`
  - `--dagmm-lambda-1`: coefficient for the energy loss
    - Default: `0.1`
  - `--dagmm-lambda-2`: coefficient for the penalization of degenerate covariance matrices
    - Default: `0.005`
  - `--dagmm-reg-covar`: small epsilon value added to covariance matrix to ensure it remains invertible
  - `--dagmm-compression-factor`: compression factor for the AE network
    - Default: `2`
  - `--dagmm-act-fn`: activation function of the AD network
    - Default: `relu`
  - `--gmm-act-fn`: activation function of the GMM network
    - Default: `tanh`
- SOM-DAGMM
  - `--somdagmm-n-soms`: number of soms
    - Default: `1`
  - `--somdagmm-latent-dim`: latent dimension of the AE subnetwork
  - Default: `1`
  - `--somdagmm-n-mixtures`: number of mixtures for the GMM network
    - Default: `4`
  - `--somdagmm-n-layers`: number of layers of the AE network
    - Default: `4`
  - `--somdagmm-lambda-1`: coefficient for the energy loss
    - Default: `0.1`
  - `--somdagmm-lambda-2`: coefficient for the penalization of degenerate covariance matrices
    - Default: `0.005`
  - `--somdagmm-reg-covar`: small epsilon value added to covariance matrix to ensure it remains invertible
  - `--somdagmm-compression-factor`: compression factor for the AE network
    - Default: `2`
  - `--somdagmm-ae-act-fn`: activation function of the AE network
    - Default: `relu`
  - `--somdagmm-gmm-act-fn`: activation function of the GMM network
    - Default: `tanh`
- MemAE
  - `--memae-shrink_thres`: threshold for hard shrinking relu
    - Default: `0.0025`
  - `--memae-latent-dim`: latent dimension of the AE network
    - Default: `1`
  - `--memae-mem-dim`: number of memory units
    - Default: `50`
  - `--mem-ae-alpha`: coefficient for the entropy loss
  - `--memae-n-layers`: number of layers of the AE network
    - Default: `4`
  - `--memae-compression-factor`: compression factor for the AE network
    - Default: `2`
  - `--somdagmm-ae-act-fn`: activation function of the AE network
    - Default: `relu`
- DeepSVDD
  - `--deepsvdd-n-layers`: number of layers
    - Default: `2`
  - `--deepsvdd-compression-factor`: compression factor of the network
    - Default: `4`
  - `--deepsvdd-act-fn`: activation function of the network
    - Default: `relu`
- DROCC
  - `--drocc-lamb`: weight given to the adversarial loss
    - Default: `1.`
  - `--drocc-radius`: radius of hypersphere to sample points from
    - Default: `3.`
  - `--drocc-gamma`: parameter to vary projection
    - Default: `2.`

## Suggested parameters
#### DAGMM
| Dataset    | n_mixtures | latent_dim | lambda_1 | lambda_2 | reg_covar |
|------------|------------|------------|----------|----------|-----------|
| Arrhythmia | 2          | 1          | 0.1      | 0.005    | 1e-12     |
| Thyroid    | 4          | 1          | 0.1      | 0.005    | 1e-12     |
| Default    | 4          | 1          | 0.1      | 0.005    | 1e-12     |
 
#### AutoEncoder
| Dataset    | latent_dim | compression_factor | n_layers | act_fn |
|------------|------------|--------------------|----------|--------|
| Arrhythmia | 10         | 1                  | 2        | tanh   |
| Thyroid    | 1          | 2                  | 3        | tanh   |
| Default    | 1          | 0.005              | 4        | relu   |

#### MemAE
| Dataset    | latent_dim | compression_factor | n_layers | alpha | act_fn |
|------------|------------|--------------------|----------|-------|--------|
| Arrhythmia | 10         | 2                  | 4        | 2e-4  | tanh   |
| Thyroid    | 4          | 2                  | 3        | 2e-4  | tanh   |
| Default    | 3          | 2                  | 4        | 2e-4  | relu   |

#### SOM-DAGMM
| Dataset    | latent_dim | compression_factor | n_layers | act_fn |
|------------|------------|--------------------|----------|--------|
| Arrhythmia | ?          | ?                  | ?        | ?      |
| Thyroid    | ?          | ?                  | ?        | ?      |
| Default    | ?          | ?                  | ?        | ?      |





## Example
To train a DAGMM on the KDD 10 percent dataset with the default parameters described in the original paper:
```
$ python  -m src.main -m DAGMM -d [/path/to/dataset.npz] --dataset KDD10 --batch-size 1024 --results-path ./results/KDD10 --models-path ./models/KDD10
```
Replace `[/path/to/dataset.npz]` with the path to the dataset in a numpy-friendly format.

Optionally, a Jupyter notebook is made available in `experiments.ipynb`
