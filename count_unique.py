import pandas as pd
import re
from logscan.logscan import regex_tokenizer, is_word

def get_unique(file_path):
    print("Reading", file_path)
    # The format might be different, let's just do a rough clean
    logs = pd.read_csv(file_path)['Content']
    print(f"Total rows: {len(logs)}")
    
    clean_logs = set()
    for raw_log in logs:
        log_tokens = regex_tokenizer.tokenize(str(raw_log))
        clean_text = [t for t in log_tokens if is_word(t)]
        clean_logs.add(' '.join(clean_text))
        
    print(f"Unique clean logs: {len(clean_logs)}")

get_unique("full_dataset/HealthApp/HealthApp_full.log_structured.csv")
