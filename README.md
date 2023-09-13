# FedMoP: Federated Learning using Model Projection for Multi‑Center Disease Diagnosis with Non-IID data (Under review)

This is the PyTorch implementation of our paper **"Jie Du, Wei Li, Peng Liu, et al. Federated Learning using Model Projection for Multi-Center Disease Diagnosis with Non-IID Data"** . 

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

The experimental results (test accuracy %) on COVID-19 and PBC dataset.

<table>
        <tr>
            <th colspan="2" rowspan="2">Methods</th><th colspan="2">Non-uniform</th><th colspan="2">One-class </th>
        </tr>
        <tr>
            <td>COVID-19</td><td align="center">PBC</td><td>COVID-19</td><td align="center">PBC</td>
        </tr>
        <tr>
            <td colspan="2" align="center">Centralized learning</td><td colspan="1">68.60±0.05</td><td colspan="1">97.45±0.03</td><td colspan="1">68.60±0.05</td><td colspan="1">97.45±0.03</td>
        </tr>
            <tr>
                <td rowspan="6">Federated learning </td><td>FedAvg(AISat-2017)</td><td >66.31±0.24</td><td>96.74±0.06</td><td>67.05±0.16</td><td>87.22±0.29</td>
            </tr>
            <tr>
                <td>FedProx(PMLS-2020)</td></td><td>66.41±0.32</td><td>96.78±0.10</td><td>66.89±0.32</td><td>87.88±0.15</td>
            </tr>
            <tr>
                <td>SCAFFOLD(ICML-2020)</td></td><td>68.44±0.12</td><td>96.83±0.02</td><td>66.76±0.65</td><td>69.79±0.51</td>
            </tr>
            <tr>
                <td>FedaGrac(TPDS-2023)</td></td><td>68.60±0.49</td><td>96.98±0.05</td><td>68.18±0.25</td><td>88.80±0.16</td>
            </tr>
            <tr>
                <td>FedReg(ICLR-2022)</td></td><td>68.09±1.22</td><td>96.81±0.04</td><td>69.66±0.62</td><td>91.93±0.36</td>
            </tr>
            <tr>
                <td>Our FedMoP</td></td><td><b><b>70.30±0.30</b> (↑1.70)</td><td><b>97.48±0.08</b> (↑0.50)</td><td><b>70.88±0.40</b> (↑1.22)</td><td><b>97.24±0.07</b> (↑5.31)</td>
            </tr>
    
</table>



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


