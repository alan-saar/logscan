# Logscan
An automatic log parser tool based on the DBSCAN algorithm.

## Overview
Logscan uses Term Frequency-Inverse Document Frequency (TF-IDF) and Density-Based Spatial Clustering of Applications with Noise (DBSCAN) to automatically parse log messages and extract templates.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Loghub 2k Benchmark (Standard)
To run the benchmark on the original Loghub 2k dataset:

```bash
python3 -m logscan
```
This will process datasets defined in `logscan.py` (e.g., HDFS, Hadoop, Spark, Linux, etc.) located in `logs/loghub_2k/` and save results to `Logscan_result/`.

### Run Loghub 2.0 Benchmark (New)
To run the benchmark on the new Loghub 2.0 dataset (currently supporting Linux logs):

```bash
python3 -m logscan --v2
```
This will process the Linux dataset located in `full_dataset/` and save results to `Logscan_loghub2_results/`.

### Run Quick Test
To run a quick test on a small sample (Android 2k):

```bash
python3 -m logscan --test
```

## Comparison Results
A Jupyter Notebook is provided to compare the parsing accuracy between Loghub 2k and Loghub 2.0 datasets.

### Running the Comparison
1. Ensure both benchmarks have been run (standard and --v2).
2. Execute the notebook:
   ```bash
   jupyter notebook compare_results.ipynb
   ```
   Or run it from the command line:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace compare_results.ipynb
   ```

### Visualization
Open `compare_results.ipynb` to view the bar chart comparing the accuracy scores.

## Data
- **Loghub 2k**: Located in `logs/loghub_2k/`.
- **Loghub 2.0**: Located in `full_dataset/`.
- **Test Sample**: `logs/Andriod_2k.log_structured.csv` provided by [logparser](https://github.com/logpai/logparser).
