# Benchmarking the Performance of Tabular Data Generation

This repository contains the complete implementation and scripts for my research project **"Benchmarking the Performance of Tabular Data Generation"**, developed and executed in **Python 3.9**. The goal of this project is to evaluate and compare the performance of five state-of-the-art tabular data generation models using a unified benchmarking pipeline.

## 🔧 Environment Setup

The project uses Conda environments (`env_tf.yml` and `env_torch.yml`) to manage dependencies for TensorFlow and PyTorch-based models respectively.
	Code for preprocessing dataset directly downloaded from the UCI repository (assuming the target column is the last one; adjust if needed). Constant value columns are discarded to avoid affecting accuracy.
python Scripts\preprocess_discrete.py --input Raw\adult.csv --output Discrete\adult_discrete.csv

For datasets in datasets_Dm folder
Code for converting arff to csv file and encode discrete numeric columns in it are given at the end of the file.
	After that, use below code for preprocessing these datasets

python Scripts\preprocess_encode_only.py --input Raw\adult.csv --output Discrete\adult_discrete.csv

	Code for splitting dataset to train and test 
python Scripts\split_dataset.py   --input_csv Discrete\adult_discrete.csv  --output_dir Data\adult –seed 42
Ganblr
For creating conda env for ganblr,
conda env create -f env_tf.yml 
conda activate tabgen-tf
	Code for training ganblr model is given below. For example, 
python Scripts/ganblr_train.py --dataset adult --size_category medium
	For TSTR evaluation
python Scripts\tstr_evaluation.py --synthetic_dir Synthetic/adult/ganblr --real_test_dir Data/adult

For running other models, create a conda env to use pytorch
conda env create -f env_torch.yml 
conda activate tabgen-torch
Ctabgan-plus
	Code for training ctabganplus model is given below. For example, 
python Scripts\ctabganplus_train.py   --dataset_name adult  --size_category medium
	For TSTR evaluation
python Scripts\tstr_evaluation.py --synthetic_dir Synthetic/adult/ctabgan_plus --real_test_dir Data/adult
tabddpm
	Code for training tabddpm model is given below. For example, 
python Scripts\tabddpm_train.py --dataset adult

	For TSTR evaluation
python Scripts\tstr_evaluation.py --synthetic_dir Synthetic/adult/tabddpm --real_test_dir Data/adult
tabsyn
	we need to create npy files for running tabsyn
python Scripts/create_npy.py --dataset adult
	Code for training tabsyn model is given below. For example, 
python tabsyn\vae\main.py  --dataname adult --gpu 0
python Scripts\tabsyn_train.py --dataset adult
	For TSTR evaluation
python Scripts\tstr_evaluation.py --synthetic_dir Synthetic/adult/tabsyn --real_test_dir Data/adult
Great 
	Code for training great model is given below. For example, 
python Scripts\great_train.py --dataset adult
	For TSTR evaluation
python Scripts\tstr_evaluation.py --synthetic_dir Synthetic/adult/great --real_test_dir Data/adult

Code for processing arff files in datasets_Dm folder
import re
import pandas as pd
from scipy.io import arff

# === 1. Load ARFF File ===
data, meta = arff.loadarff('adult.arff')
df = pd.DataFrame(data)

# === 2. Decode Byte Strings
df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# === 3. Clean: remove \, ", and ' from strings
df = df.applymap(lambda x: re.sub(r'[\\\'\"]', '', x) if isinstance(x, str) else x)

# === 4. Encode pre-discretized interval bins into 0-n integer labels ===
def extract_lower_bound(interval):
    match = re.match(r"\(?(-?[\d\.inf]+)-", interval)
    if match:
        val = match.group(1)
        return float('-inf') if val == '-inf' else float(val)
    return float('inf')  # fallback if format doesn't match

def encode_bins_numerically(series):
    unique_bins = series.dropna().unique()
    sorted_bins = sorted(unique_bins, key=extract_lower_bound)
    bin_to_id = {bin_val: i for i, bin_val in enumerate(sorted_bins)}
    return series.map(bin_to_id), bin_to_id

# === 5. Apply encoding ONLY to already binned columns ===
# give the column names of binned columns in binned_cols
binned_cols = [ ]  
for col in binned_cols:
    df[col], mapping = encode_bins_numerically(df[col])
    print(f"\nMapping for {col}:")
    for k, v in mapping.items():
        print(f"{k} → {v}")

# === 6. Save final CSV ===
df.to_csv("adult.csv", index=False)
print("\nSaved cleaned dataset to 'adult.csv'")
