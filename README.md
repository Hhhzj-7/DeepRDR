# DeepRDR

DeepRDR: ncRNA-Drug Resistance Prediction via Cross-Context Contrastive Learning and Gated Cross Fusion

---

## Directory Structure

```
DeepRDR/
├── cold_start_splits
│   ├── all_cold_start_splits.csv      # All cold-start split definitions in CSV format
│   ├── all_cold_start_splits.pkl      # Cold-start split data
│   ├── both_all_folds.csv             # Cross-validation folds where both drug and RNA are unseen
│   ├── both_splits.pkl                # Splits for both cold-start scenario
│   ├── drug_all_folds.csv             # Fold information for drug cold-start experiments
│   ├── drug_splits.pkl                # Drug cold-start splits
│   ├── rna_all_folds.csv              # Fold information for RNA cold-start experiments
│   └── rna_splits.pkl                 # RNA cold-start splits
│
├── Data
│   ├── adj_with_sens.csv              # Adjacency matrix including resistance/sensitive labels
│   ├── DrugCentral_approved_Drugs.csv # List of approved drugs from DrugCentral database
│   ├── drug_smiles.csv                # Drug SMILES representations
│   ├── fold_info.pickle               # Precomputed cross-validation fold information
│   ├── HNC_RNA_case.csv               # RNA case study dataset (HNC-related RNAs)
│   ├── independent_data.csv           # Independent test dataset
│   ├── ncrna-drug_split.csv           # Dataset split information for ncRNA–drug interactions
│   ├── processed_data_cache.pkl       # Cached processed dataset for faster loading
│   └── rna_sequences.fasta            # RNA sequence data in FASTA format
│
├── main.py                           # Standard setting
├── main_cold.py                      # Cold-start setting
├── main_independent.py               # Independent setting
│
├── models.py                         # Model architecture definitions
├── utils.py                          # Utility functions (data processing, etc.)
│
├── standard_fold                     # Directory for saving results of standard cross-validation
├── save                              # Directory for storing standard setting results
├── save_cold                         # Directory for storing cold-start setting results

```

---

## Environment

Create a new conda environment:

```bash
conda create -n deeprdr python=3.8
conda activate deeprdr
pip install -r requirements.txt
```

---

## Experiments

Three main experimental setups are implemented:

### 1. Standard setting

Run:

```bash
python main.py
```

* Outputs to `save/`

### 2. Cold-start setting

Run:

```bash
python main_cold.py
```
* Outputs to `save_cold/`

### 3. Independent setting

Run:

```bash
python main_independent.py
```

---

## Contact

If you have any issues or questions about this paper or need assistance with reproducing the results, please contact me.

Zhijian Huang

School of Computer Science and Engineering,

Central South University

Email: zhijianhuang@csu.edu.cn
