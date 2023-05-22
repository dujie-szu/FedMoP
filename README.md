# FedMoP: Federated Learning using Model Projection for Multi‑Center Disease Diagnosis with Non-IID data

This is the PyTorch implementation of our paper **"Federated Learning using Model Projection for Multi-Center Disease Diagnosis with Non-IID Data"** . 


## Usage

#### requirements：
- Ubuntu Server == 20.04.4 LTS
- CUDA == 11.6
- numpy ==1.23.1
- Pillow == 9.2.0
- python == 3.8.0
- quadprog == 0.1.11
- torch == 1.12.0
- torchvision == 0.13.0

### Datasets：

- **HAM10k**：Please manually download the [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) dataset from the official website, unzip it and place it in './dataset/ham10k'.
- **COVID-19**：Please manually download the [COVID-19 Five-classes](https://www.kaggle.com/datasets/edoardovantaggiato/covid19-xray-two-proposed-databases?select=Datasets)  dataset from the official website, unzip it  and place it in './dataset/covid19'.
- **PBC**:  Please manually download the [PBC](https://data.mendeley.com/datasets/snkd93bnjr/1) dataset from the official website, unzip it  and place it in './dataset/pbc'.



#### Split the dataset in feature/label distribution skew：

You can use `split_dataset.py`  to split the dataset into feature distribution skew and label distribution skew, and then save the split data set for model training.


### Training:

`main.py` is the main file to run the federated experiments.

The experiments can be run by:

```
cd FedMoP
python main.py
```


