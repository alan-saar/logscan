# Logscan
An automatic log parser tool based on the DBSCAN algorithm.

## Installation

```
pip install -r requirements.txt
```

## Usage

### Logscan (Loghub 2k Dataset)
To run logscan on the Loghub 2k dataset:

1. Ensure your Loghub 2k data is in the `logs/loghub_2k/` directory.
2. Run the module:
   ```bash
   python -m logscan
   ```
3. Results will be saved in `Logscan_result/`.

### Logscan-LLM (Loghub 2.0 Dataset)
To run logscan-llm on the updated Loghub 2.0 dataset:

1. Ensure your Loghub 2.0 data is in the `full_dataset/` directory (e.g., `full_dataset/Linux/Linux_full.log_structured.csv`).
2. Run the module:
   ```bash
   python -m logscan-llm
   ```
3. Results will be saved in `Logscan-llm_result/`.

## Data
The dataset Andriod_2k.log_structured.csv used as a test was provided by [logparser](https://github.com/logpai/logparser).
