# FedMoP: Federated Learning using Model Projection for Multi‑Center Disease Diagnosis with Non-IID data (Under review)

This is the PyTorch implementation of our paper **"Federated Learning using Model Projection for Multi-Center Disease Diagnosis with Non-IID Data"** . 

The experimental  results (test accuracy %) on HAM10K dataset.

|  Methods | Two-client | Four-client  |
|---|---|---|
|  Centralized learning | 68.94±0.06  | 68.94±0.06  |
|  FedAvg(AISat-2017) |  65.43±0.29 | 59.22±0.77  |
|  FedProx(PMLS-2020) |  66.25±0.29 | 59.22±0.29  |
|  SCAFFOLD(ICML-2020) | 67.08±1.34  | 64.39±0.29  |
| FedaGrac(TPDS-2023)  | 67.70±0.88  |  64.18±0.59 |
|  FedReg(ICLR-2022) | 66.67±1.05  |  65.84±0.05 |
|  Our FedMoP | 69.36±0.30 (↑1.66) | 69.57±0.51 (↑3.73)  |

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


