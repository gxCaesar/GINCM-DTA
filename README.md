# GINCM-DTA: A transferable deep COVID-19 multi-targeted drug repurposing protocol for screening broad-spectrum antivirals
GINCM-DTA is a novel transfer-learnable graph isomorphism network with protein contact map for drug-target affinity prediction. GINCM-DTA is pre-trained on Davis and KIBA datasets and then fine-tuned on COVID-DTA through knowledge transfer. Finally, the fine-tuned GINCM-DTA is used to screen potential anti-SARS-CoV-2 drug candidates from FDA-approved drugs. 

GINCM-DTA includes a graph representation module and a prediction module. First, the molecular graph representation was obtained by the SMILES treatment of the drug with RDKit. Meanwhile, we first processed the protein sequence into contact map by PconsC4, and then obtained the protein graph representation via the position frequency matrix. After that, we obtain DTA predictions through a welldesigned GIN framework.

## Reproducibility
The analysis in the paper can be completely reproduced. You can reproduce or train on your own data. You may need the directions below to reproduce all results correctly. Also, if you want to leverage the GINCM-DTA protocol to repurpose drugs for COVID-19 or other diseases, or if you want to retrain the model, please check out the corresponding section in this README.

## Installation & Dependencies
The code of GINCM-DTA is written in Python 3, which is mainly tested on Python 3.7 and Linux OS. It's faster to train on a GPU, but you can also work on a standard computer.

GINCM-DTA has the following dependencies:
* hhsuite (https://github.com/soedinglab/hh-suite)
* ccmpred (https://github.com/soedinglab/CCMpred)
* torch==1.12.1
* PyG (torch-geometric) == 1.3.2
* scikit-learn==0.24.2
* matplotlib==3.4.3
* numpy==1.20.3
* pandas==1.3.4
* networkx==2.6.3
* tqdm==4.62.3
* scipy==1.7.1
* rdkit==2021.09.2
* transformers==4.12.2
* tensorboardx==2.4

## data preparation and preprocess
+ 1. dataset split
    + Get the DTA files of the Davis and KIBA datasets (downloaded in https://github.com/hkmztrk/DeepDTA/tree/master/data). You first need to generate the standard data and 5-fold cross-validation data from raw data. Run command to precess the rawData as follows:
    ```python
    python data_process.py
    ```
    Then, you will get the processed data files:
    - data/davis_test.csv
    - data/davis_train.csv
    - data/kiba_test.csv
    - data/kiba_train.csv
    - data/davis/davis_fold0-5_train.csv
    - data/davis/davis_fold0-5_valid.csv
    - data/kiba/kiba_fold0-5_train.csv
    - data/kiba/kiba_fold0-5_valid.csv

+ 2. contact map generation
    + You need to use Pconsc4 to predict the contact map through the protein sequence of all targets in the Davis and KIBA datasets. Perform all steps by:
    ```python
    python cm_generation.py
    ```
    Then, you will get the processd data files. Then copy the two resulting folders named "aln" and "pconsc4" from to the data folder. The contact maps of the targets of the Davis, KIBA and covid datasets can all be obtained by executing the cm_generation.py file.

## training in davis or KIBA
To train the model in davis or kiba datasets:

```python
python train.py --batch_size 256 --epochs 2000 --lr 0.001 --dataset davis
python train.py --batch_size 256 --epochs 2000 --lr 0.001 --dataset kiba
```

## Fine-tuning in COVID-19 dataset
After training the model in davis or kiba datasets, you will get a model parameter file in '\model' folder.
Then, you can finetune the model in COVID-19  dataset:
```python
python finetune.py --batch_size 16 --epochs 2000 --lr 0.001 --load_model_path ./model/default.pt
```

##  Contact
Please contact chengx48@mail2.sysu.edu.cn for help or submit an issue.
