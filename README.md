# FedMoP: Federated Learning using Model Projection for Multi‑Center Disease Diagnosis with Non-IID data (Under review)

This is the PyTorch implementation of our paper **"Federated Learning using Model Projection for Multi-Center Disease Diagnosis with Non-IID Data"** . 

The experimental  results (test accuracy %) on HAM10K dataset.

<table>
    <tr>
        <th colspan="2">Methods</th><th>Two-client</th><th>Four-client </th>
    </tr>
  <tr>
    <td colspan="2">Centralized learning</td><td colspan="2" align="center">68.94±0.06</td>
  </tr>
    <tr>
        <td rowspan="6">Federated learning </td><td>FedAvg(AISat-2017)</td><td >65.43±0.29 </td><td>59.22±0.77</td>
    </tr>
    <tr>
        <td>FedProx(PMLS-2020)</td></td><td>66.25±0.29</td><td>59.22±0.29</td>
    </tr>
    <tr>
        <td>SCAFFOLD(ICML-2020)</td></td><td>67.08±1.34</td><td>64.39±0.29</td>
    </tr>
    <tr>
        <td>FedaGrac(TPDS-2023)</td></td><td>67.70±0.88</td><td>64.18±0.59</td>
    </tr>
    <tr>
        <td>FedReg(ICLR-2022)</td></td><td>66.67±1.05</td><td>65.84±0.05</td>
    </tr>
    <tr>
        <td>Our FedMoP</td></td><td><b>69.36±0.30</b> (↑ 1.66)</td><td><b>69.57±0.51</b> (↑ 3.73)</td>
    </tr>
   
</table>

作者：图南
链接：https://www.zhihu.com/question/50267650/answer/1584380105
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。




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


