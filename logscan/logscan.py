
"""
LogScan Module.

This module implements the LogScan algorithm for automated log parsing and template extraction.
It uses a pipeline of text preprocessing, TF-IDF vectorization, DBSCAN clustering,
and word frequency analysis to identify variable parts of log messages.
"""

from . import __version__

# Libs
import pandas as pd
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import operator
import numpy as np
import sys
import os
sys.path.append("../")

nltk.download('punkt')
regex_tokenizer = RegexpTokenizer(r'\w+')

from .auxiliares import is_word, replace_space, has_numbers, word_position, word_counter, remove_repeated
# Funções auxiliares

def log_template(cluster_tagger_dict, log):
  """
  Generates a log template by replacing variable parts with '<*>'.

  Args:
      cluster_tagger_dict (dict): Dictionary mapping token to (variable/template) label.
      log (str): The raw log message.

  Returns:
      tuple: (template_string, list_of_variables)
  """
  new_log = replace_space(log)
  tokens = wordpunct_tokenize(new_log)
  variables_list = []
  template = ''
  for token in tokens:
    if token != "_IS_SPACE_":
      info = cluster_tagger_dict.get(token, 'template')
      if info == 'variable':
        variables_list.append(token)
        template = template + '<*>'
      else:
        template = template + token
    else:
      template = template + ' '
  return template, variables_list


def word_classifier(wordfrequency):
  """
  Classifies words as 'variable' or 'template' based on frequency.

  Words with frequency below the 30th percentile or containing numbers
  are classified as variables.

  Args:
      wordfrequency (list): List of (word, position, frequency) tuples.

  Returns:
      list: List of (word, position, frequency, label) tuples.
  """
  frequency_list = list(map(operator.itemgetter(2), wordfrequency))
  p30 = np.percentile(frequency_list, 30)
  label = []
  for word in wordfrequency:
    if word[2] < p30 or has_numbers(word[0]):
      # word[0] -> token
      # word[1] -> posicao
      # word[2] -> quantidade
      label.append((word[0],word[1],word[2],"variable"))
    else:
      label. append((word[0],word[1],word[2],"template"))
  return label

# Logscan

class LogScan:
  """
  Main class for the LogScan algorithm.

  Attributes:
      data (pd.DataFrame): DataFrame storing log data and processing results.
  """
  def _update_progress(self, percent, step_name=None):
    if step_name is not None:
        self.current_step = step_name
    bars = int(percent / 5)
    if bars > 20: bars = 20
    if percent > 100: percent = 100
    bar_str = '#' * bars + ' ' * (20 - bars)
    step_str = f" - {self.current_step}" if hasattr(self, 'current_step') else ""
    out = f'\r[{bar_str}] {percent}%{step_str}'
    sys.stdout.write(out.ljust(80))
    sys.stdout.flush()

  def __init__(self, logdata: list, header: bool, header_regex = None, regex_list = None, test_n = None):
    """
    Initializes the LogScan instance.

    Args:
        logdata (list): List of raw log strings.
        header (bool): Whether to strip headers using regex.
        header_regex (str, optional): Regex pattern for header removal.
        regex_list (list, optional): List of generic regex strings to mask variables before clustering.
        test_n (int, optional): Number of rows to print per execution pipeline step.
    """
    # print('- Logscan v1.0')
    self._update_progress(0, "Inicializando")
    self.test_n = test_n
    self.test_output = ""
    
    if regex_list:
        processed_data = []
        for log in logdata:
            for rgx in regex_list:
                log = re.sub(rgx, '<*>', log)
            processed_data.append(log)
        logdata = processed_data
        
    if header:
      # print('-- Header Extraction')
      loglist= [re.sub(f'{header_regex}', '', log) for log in logdata]
      self.data = pd.DataFrame(loglist,columns=['Log'])
    else:
      self.data = pd.DataFrame(logdata,columns=['Log'])

    if self.test_n is not None:
        self.test_output += "Input:\n"
        for log in logdata[:self.test_n]:
            self.test_output += str(log) + "\n"

  def clean_data(self):
    """
    Preprocesses log data by tokenizing and removing non-word characters.

    Populates the 'CleanLog' column in self.data.
    """
    # print('-- Data Cleaning')
    self._update_progress(0, "Data Cleaning")
    clear_content = []
    total = len(self.data)
    for i, (_, row) in enumerate(self.data.iterrows()):
        self._update_progress(int(i / max(1, total) * 20))
        raw_log = row['Log']
        log_tokens = regex_tokenizer.tokenize(raw_log)
        clean_text = []
        for token in log_tokens:
            if is_word(token):
                clean_text.append(token)
        clean_log = ' '.join(clean_text)
        clear_content.append(clean_log)
    self.data['CleanLog'] = clear_content
    
    if self.test_n is not None:
        self.test_output += "\nTratamento dos dados:\n"
        for log in self.data['CleanLog'].head(self.test_n):
            self.test_output += str(log) + "\n"

  def tfidf_transformer(self):
    """
    Converts unique cleaned logs to TF-IDF vectors.

    Returns:
        tuple: (vectors, unique_clean_logs)
    """
    # print('-- TF-IDF Transformer')
    self._update_progress(20, "TF-IDF Transformer")
    unique_clean_logs = self.data['CleanLog'].drop_duplicates().reset_index(drop=True)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(unique_clean_logs)
    
    if self.test_n is not None:
        self.test_output += "\nTF-IDF:\n"
        test_tfidf = vectorizer.transform(self.data['CleanLog'].head(self.test_n))
        for i in range(min(self.test_n, test_tfidf.shape[0])):
            row = test_tfidf.getrow(i)
            self.test_output += f"Log {i} TF-IDF: {row}\n"
            
    return vectors, unique_clean_logs

  def dbscanModel(self, vectors, unique_clean_logs):
    """
    Apply DBSCAN clustering to the log embeddings.

    Populates the 'Cluster' column in self.data.

    Args:
        vectors (scipy.sparse.csr_matrix): TF-IDF feature matrix.
        unique_clean_logs (pd.Series): The unique logs.
    """
    # print('-- DBSCAN')
    self._update_progress(30, "DBSCAN Clustering")
    clusterModel = DBSCAN(min_samples=2)
    clusterModel.fit(vectors)
    
    # Map clusters back to all rows based on their CleanLog
    cluster_map = dict(zip(unique_clean_logs, clusterModel.labels_))
    self.data['Cluster'] = self.data['CleanLog'].map(cluster_map)
    
    if self.test_n is not None:
        self.test_output += "\nDBSCAN:\n"
        for log, cl in zip(self.data['CleanLog'].head(self.test_n), self.data['Cluster'].head(self.test_n)):
            self.test_output += f"Cluster: {cl} | CleanLog: {log}\n"

  def word_tagger(self):
    """
    Analyzes word frequency per cluster to tag words as variables or template parts.

    Returns:
        list: A list of (cluster_id, labeled_words) tuples.
    """
    # print('-- Word Tagger')
    self._update_progress(40, "Word Tagger")
    tagger = []
    clusters = np.unique(self.data['Cluster'])
    total_clusters = len(clusters)
    for i, cluster in enumerate(clusters):
      self._update_progress(40 + int((i / max(1, total_clusters)) * 30))
      # Count occurrences of each unique log in this cluster
      dados_cluster = self.data.loc[self.data['Cluster'] == cluster]['Log']
      unique_logs_counts = dados_cluster.value_counts()
      
      from collections import Counter
      cluster_token_counts = Counter()
      
      for log, count in unique_logs_counts.items():
        tokens = wordpunct_tokenize(log)
        tokens_position = word_position(tokens)
        # Update counter with multiplied counts
        for tp in tokens_position:
            cluster_token_counts[tp] += count
            
      # Convert counter back to word_frequency format
      new_wordfrequency = [(w[0], w[1], count) for w, count in cluster_token_counts.items()]
      wordlabel = word_classifier(new_wordfrequency)
      tagger.append((cluster, wordlabel))
      
    if hasattr(self, 'test_n') and self.test_n is not None:
        self.test_output += "\nLabels / Tagger:\n"
        first_n_clusters = self.data['Cluster'].head(self.test_n).values
        tagger_dict = dict(tagger)
        for i, cl in enumerate(first_n_clusters):
            labels = tagger_dict.get(cl, [])
            labels_str = ", ".join([f"('{w[0]}', freq:{w[2]}, {w[3]})" for w in labels])
            self.test_output += f"Log {i} (Cluster {cl}) Tagger: {labels_str}\n"
            
    return tagger

  def create_templates(self, tagger):
    """
    Generates templates for all logs based on the tagger results.

    Populates 'Template' and 'Variables' columns in self.data.

    Args:
        tagger (list): Output from word_tagger().
    """
    # print('-- Template Extraction')
    self._update_progress(70, "Template Extraction")
    templates = []
    variables = []
    
    # Convert tagger to a dictionary for faster lookups
    tagger_dict = {}
    for item in tagger:
        word_dict = {}
        for w in item[1]:
            if w[0] not in word_dict:
                word_dict[w[0]] = w[3]
        tagger_dict[item[0]] = word_dict
    
    total = len(self.data)
    memo = {}
    for i, (index, row) in enumerate(self.data.iterrows()):
      self._update_progress(70 + int((i / max(1, total)) * 30))
      log_str = row['Log']
      log_cluster = row['Cluster']
      
      if log_str in memo:
          template, variables_list = memo[log_str]
      else:
          log_tagger = tagger_dict.get(log_cluster)
          template, variables_list = log_template(log_tagger, log_str)
          memo[log_str] = (template, variables_list)
          
      templates.append(template)
      variables.append(variables_list)
      
    self.data['Template'] = templates
    self.data['Variables'] = variables

  def pipeline(self):
    """
    Runs the full LogScan pipeline.

    Returns:
        tuple: (tagger, result_dataframe)
    """
    try:
        self.clean_data()
        vectors, unique_clean_logs = self.tfidf_transformer()
        self.dbscanModel(vectors, unique_clean_logs)
        tagger = self.word_tagger()
        self.create_templates(tagger)
        self._update_progress(100, "Concluído")
        print()
        if hasattr(self, 'test_n') and self.test_n is not None:
            print("\n" + self.test_output.strip())
            
        return tagger, self.data
    except Exception as e:
        print()
        if hasattr(self, 'test_n') and self.test_n is not None and hasattr(self, 'test_output'):
            print("\n[ERRO NA PIPELINE] Intermediate Logs:")
            print(self.test_output.strip())
        raise e


# Main logic handled below



def original_parsing_accuracy(data):
    log_per_template =  data['EventId'].value_counts().to_dict()
    correct = 0
    for cluster in np.unique(data['Cluster']):
        data_cluster = data.loc[data['Cluster'] == cluster]
        log_per_template_cluster =  data_cluster['EventId'].value_counts().to_dict()
        for eventid in np.unique(data_cluster['EventId']):
            if log_per_template[eventid] == log_per_template_cluster[eventid]:
                correct = correct + log_per_template_cluster[eventid]
    return correct/len(data)

def parsing_accuracy(data):
    if 'EventTemplate' not in data.columns or 'Template' not in data.columns:
        raise ValueError("Both 'EventTemplate' and 'Template' columns must be present in data for exact PA.")
    
    correct = 0
    for idx, row in data.iterrows():
        gen = str(row['Template']).strip()
        ref = str(row['EventTemplate']).strip()
        
        gen = re.sub(r'\s+', ' ', gen)
        ref = re.sub(r'\s+', ' ', ref)
        
        if gen == ref:
            correct += 1
            
    return correct/len(data)

def template_accuracy(data):
    if 'EventTemplate' not in data.columns or 'Template' not in data.columns:
        raise ValueError("Both 'EventTemplate' and 'Template' columns must be present in data for FTA.")
        
    parsed_templates = set()
    oracle_templates = set()
    
    for idx, row in data.iterrows():
        gen = str(row['Template']).strip()
        ref = str(row['EventTemplate']).strip()
        
        gen = re.sub(r'\s+', ' ', gen)
        ref = re.sub(r'\s+', ' ', ref)
        
        parsed_templates.add(gen)
        oracle_templates.add(ref)
        
    correct_templates = parsed_templates.intersection(oracle_templates)
    
    pta = len(correct_templates) / len(parsed_templates) if len(parsed_templates) > 0 else 0
    rta = len(correct_templates) / len(oracle_templates) if len(oracle_templates) > 0 else 0
    
    if pta + rta == 0:
        fta = 0.0
    else:
        fta = 2 * (pta * rta) / (pta + rta)
        
    return fta, pta, rta

def grouping_accuracy(data):
    if 'EventId' not in data.columns or 'Cluster' not in data.columns:
        raise ValueError("Both 'EventId' and 'Cluster' columns must be present in data for GA and FGA.")

    oracle_groups_map = data.groupby('EventId').groups
    parsed_groups_map = data.groupby('Cluster').groups
    
    oracle_groups = set(frozenset(indices) for indices in oracle_groups_map.values())
    parsed_groups = set(frozenset(indices) for indices in parsed_groups_map.values())
    
    correct_groups = parsed_groups.intersection(oracle_groups)
    
    ga = sum(len(g) for g in correct_groups) / len(data) if len(data) > 0 else 0
    
    pga = len(correct_groups) / len(parsed_groups) if len(parsed_groups) > 0 else 0
    rga = len(correct_groups) / len(oracle_groups) if len(oracle_groups) > 0 else 0
    
    if pga + rga == 0:
        fga = 0.0
    else:
        fga = 2 * (pga * rga) / (pga + rga)
        
    return ga, fga, pga, rga

import argparse
import time

def run_benchmark(input_dir, output_dir, settings, result_file="Logscan_benchmark_result.csv", eval_all=False, original_pa=False, test_n=None, correct_pa=None, incorrect_pa=None):
    total_start_time = time.time()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    result_dir = os.path.dirname(result_file)
    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir)

    benchmark_result = []
    for dataset, setting in settings.items():
        dataset_start_time = time.time()
        print(f"\n=== Avaliando: {dataset} ===", flush=True)
        indir = os.path.join(input_dir, os.path.dirname(setting["log_file"]))
        log_file = os.path.basename(setting["log_file"])

        try:
            test_dataset = pd.read_csv(os.path.join(indir, log_file + "_structured.csv"))
            regex_list = setting.get("regex", [])
            log_scan_dataset = LogScan(list(test_dataset['Content']), header=False, regex_list=regex_list, test_n=test_n)
            tagger, result_dataset = log_scan_dataset.pipeline()
            result_dataset['EventId'] = test_dataset['EventId']
            if 'EventTemplate' in test_dataset.columns:
                result_dataset['EventTemplate'] = test_dataset['EventTemplate']
            result_dataset.to_csv(os.path.join(output_dir, log_file + "_structured.csv"))
            
            if test_n is not None or correct_pa is not None or incorrect_pa is not None:
                print("\nTemplate final (com ground truth):")
                
                rows_to_print = []
                log_per_template = {}
                cluster_counts_per_event = {}
                
                if original_pa and 'EventId' in result_dataset.columns:
                    log_per_template = result_dataset['EventId'].value_counts().to_dict()
                    for cluster in np.unique(result_dataset['Cluster']):
                        data_cluster = result_dataset.loc[result_dataset['Cluster'] == cluster]
                        cluster_counts_per_event[cluster] = data_cluster['EventId'].value_counts().to_dict()
                
                if correct_pa is not None or incorrect_pa is not None:
                    correct_collected = 0
                    incorrect_collected = 0
                    
                    for i in range(len(result_dataset)):
                        max_c = correct_pa if correct_pa is not None else 0
                        max_i = incorrect_pa if incorrect_pa is not None else 0
                        
                        if correct_collected >= max_c and incorrect_collected >= max_i:
                            break
                        
                        row = result_dataset.iloc[i]
                        gen = str(row.get('Template', '')).strip()
                        ref = str(row.get('EventTemplate', '')).strip()
                        gen = re.sub(r'\s+', ' ', gen)
                        ref = re.sub(r'\s+', ' ', ref)
                        is_correct_strict = (gen == ref)
                        
                        is_correct_original = False
                        if original_pa and 'EventId' in row:
                            e_id = row['EventId']
                            c_id = row['Cluster']
                            if e_id in log_per_template and c_id in cluster_counts_per_event and e_id in cluster_counts_per_event[c_id]:
                                is_correct_original = (log_per_template[e_id] == cluster_counts_per_event[c_id][e_id])
                        
                        if is_correct_strict and correct_pa is not None and correct_collected < correct_pa:
                            rows_to_print.append((row, is_correct_strict, is_correct_original))
                            correct_collected += 1
                        elif not is_correct_strict and incorrect_pa is not None and incorrect_collected < incorrect_pa:
                            rows_to_print.append((row, is_correct_strict, is_correct_original))
                            incorrect_collected += 1
                else:
                    for i in range(min(test_n, len(result_dataset))):
                        row = result_dataset.iloc[i]
                        gen = str(row.get('Template', '')).strip()
                        ref = str(row.get('EventTemplate', '')).strip()
                        gen = re.sub(r'\s+', ' ', gen)
                        ref = re.sub(r'\s+', ' ', ref)
                        is_correct_strict = (gen == ref)
                        
                        is_correct_original = False
                        if original_pa and 'EventId' in row:
                            e_id = row['EventId']
                            c_id = row['Cluster']
                            if e_id in log_per_template and c_id in cluster_counts_per_event and e_id in cluster_counts_per_event[c_id]:
                                is_correct_original = (log_per_template[e_id] == cluster_counts_per_event[c_id][e_id])
                                
                        rows_to_print.append((row, is_correct_strict, is_correct_original))
                        
                for row, is_correct_strict, is_correct_original in rows_to_print:
                    print(f"Log: {row['Log']}")
                    print(f"Template Final: {row.get('Template', '')}")
                    print(f"Ground Truth:   {row.get('EventTemplate', '')}")
                    if original_pa:
                        orig_str = str(is_correct_original).lower()
                        print(f"correct={str(is_correct_strict).lower()} (correct={orig_str} on original pa)\n")
                    else:
                        print(f"correct={str(is_correct_strict).lower()}\n")

            if original_pa:
                accuracy = original_parsing_accuracy(result_dataset)
                fta = 0.0
            else:
                accuracy = parsing_accuracy(result_dataset)
                fta, pta, rta = template_accuracy(result_dataset)
                ga, fga, pga, rga = grouping_accuracy(result_dataset)
                
            dataset_end_time = time.time()
            elapsed = dataset_end_time - dataset_start_time
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
            
            if original_pa:
                benchmark_result.append([dataset, accuracy])
            else:
                benchmark_result.append([dataset, accuracy, fta, ga, fga])
                
            print(f"=== Resultado parcial para {dataset} ===", flush=True)
            print(f"| {'Metric':<6} | {'Score':<11} |", flush=True)
            print(f"|{'-'*8}|{'-'*13}|", flush=True)
            print(f"| {'PA':<6} | {accuracy:<11.6f} |", flush=True)
            if not original_pa:
                print(f"| {'FTA':<6} | {fta:<11.6f} |", flush=True)
                print(f"| {'GA':<6} | {ga:<11.6f} |", flush=True)
                print(f"| {'FGA':<6} | {fga:<11.6f} |", flush=True)
            print(f"| {'Tempo':<6} | {time_str:<11} |", flush=True)
            print("=========================================\n", flush=True)
        except Exception as e:
            dataset_end_time = time.time()
            elapsed = dataset_end_time - dataset_start_time
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
            
            print(f"Error processing {dataset}: {e}", flush=True)
            print(f"Tempo de execução (com erro): {time_str}", flush=True)

    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    t_hours, t_rem = divmod(total_elapsed, 3600)
    t_minutes, t_seconds = divmod(t_rem, 60)
    total_time_str = f"{int(t_hours)}h {int(t_minutes)}m {t_seconds:.2f}s"

    print("\n=== Resultados ===")
    if original_pa:
        df_result = pd.DataFrame(benchmark_result, columns=["Dataset", "Accuracy"])
    else:
        df_result = pd.DataFrame(benchmark_result, columns=["Dataset", "Accuracy", "FTA", "GA", "FGA"])
    df_result.set_index("Dataset", inplace=True)
    
    dataset_exibit_order = [
        "Hadoop", "HDFS", "OpenStack", "Spark", "Zookeeper", "BGL", "HPC", 
        "Thunderbird", "Linux", "Mac", "Apache", "OpenSSH", "HealthApp", "Proxifier"
    ]
    valid_order = [d for d in dataset_exibit_order if d in df_result.index]
    valid_order += [d for d in df_result.index if d not in valid_order]
    
    df_result = df_result.reindex(valid_order)
    
    res_df = df_result.reset_index()
    if original_pa:
        print(f"{'Dataset':<15} | {'Accuracy':<10}")
        print("-" * 28)
        for _, row in res_df.iterrows():
            print(f"{row['Dataset']:<15} | {row['Accuracy']:<10.6f}")
    else:
        print(f"{'Dataset':<15} | {'Accuracy':<10} | {'FTA':<10} | {'GA':<10} | {'FGA':<10}")
        print("-" * 67)
        for _, row in res_df.iterrows():
            print(f"{row['Dataset']:<15} | {row['Accuracy']:<10.6f} | {row['FTA']:<10.6f} | {row['GA']:<10.6f} | {row['FGA']:<10.6f}")

    print(f"\nTempo total de execução do comando: {total_time_str}", flush=True)
    df_result.to_csv(result_file, float_format="%.6f")

def benchmark(original_pa=False, test_n=None, correct_pa=None, incorrect_pa=None, selected_datasets=None):
    if test_n is not None:
        input_dir = "test_dataset/v1/"
        output_dir = "results/loghub2k_test/"
    else:
        input_dir = "logs/loghub_2k/"
        output_dir = "results/loghub2k/"
    
    benchmark_settings = {
        "HDFS": {
            "log_file": "HDFS/HDFS_2k.log",
            "log_format": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
            "regex": [r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"],
            "st": 0.5,
            "depth": 4,
        },
        "Hadoop": {
            "log_file": "Hadoop/Hadoop_2k.log",
            "log_format": "<Date> <Time> <Level> \[<Process>\] <Component>: <Content>",
            "regex": [r"(\d+\.){3}\d+"],
            "st": 0.5,
            "depth": 4,
        },
        "Spark": {
            "log_file": "Spark/Spark_2k.log",
            "log_format": "<Date> <Time> <Level> <Component>: <Content>",
            "regex": [r"(\d+\.){3}\d+", r"\b[KGTM]?B\b", r"([\w-]+\.){2,}[\w-]+"],
            "st": 0.5,
            "depth": 4,
        },
        "Zookeeper": {
            "log_file": "Zookeeper/Zookeeper_2k.log",
            "log_format": "<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>",
            "regex": [r"(/|)(\d+\.){3}\d+(:\d+)?"],
            "st": 0.5,
            "depth": 4,
        },
        "OpenStack": {
            "log_file": "OpenStack/OpenStack_2k.log",
            "log_format": "<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>",
            "regex": [r"((\d+\.){3}\d+,?)+", r"/.+?\s", r"\d+"],
            "st": 0.5,
            "depth": 5,
        },
        "BGL": {
            "log_file": "BGL/BGL_2k.log",
            "log_format": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
            "regex": [r"core\.\d+"],
            "st": 0.5,
            "depth": 4,
        },
        "HPC": {
            "log_file": "HPC/HPC_2k.log",
            "log_format": "<LogId> <Node> <Component> <State> <Time> <Flag> <Content>",
            "regex": [r"=\d+"],
            "st": 0.5,
            "depth": 4,
        },
        "Thunderbird": {
            "log_file": "Thunderbird/Thunderbird_2k.log",
            "log_format": "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>",
            "regex": [r"(\d+\.){3}\d+"],
            "st": 0.5,
            "depth": 4,
        },
        "Windows": {
            "log_file": "Windows/Windows_2k.log",
            "log_format": "<Date> <Time>, <Level>                  <Component>    <Content>",
            "regex": [r"0x.*?\s"],
            "st": 0.7,
            "depth": 5,
        },
        "Linux": {
            "log_file": "Linux/Linux_2k.log",
            "log_format": "<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>",
            "regex": [r"(\d+\.){3}\d+", r"\d{2}:\d{2}:\d{2}"],
            "st": 0.39,
            "depth": 6,
        },
        "Mac": {
            "log_file": "Mac/Mac_2k.log",
            "log_format": "<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>",
            "regex": [r"([\w-]+\.){2,}[\w-]+"],
            "st": 0.7,
            "depth": 6,
        },
        "Android": {
            "log_file": "Android/Android_2k.log",
            "log_format": "<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>",
            "regex": [
                r"(/[\w-]+)+",
                r"([\w-]+\.){2,}[\w-]+",
                r"\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b",
            ],
            "st": 0.2,
            "depth": 6,
        },
        "HealthApp": {
            "log_file": "HealthApp/HealthApp_2k.log",
            "log_format": "<Time>\|<Component>\|<Pid>\|<Content>",
            "regex": [],
            "st": 0.2,
            "depth": 4,
        },
        "Apache": {
            "log_file": "Apache/Apache_2k.log",
            "log_format": "\[<Time>\] \[<Level>\] <Content>",
            "regex": [r"(\d+\.){3}\d+"],
            "st": 0.5,
            "depth": 4,
        },
        "OpenSSH": {
            "log_file": "OpenSSH/OpenSSH_2k.log",
            "log_format": "<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>",
            "regex": [r"(\d+\.){3}\d+", r"([\w-]+\.){2,}[\w-]+"],
            "st": 0.6,
            "depth": 5,
        },
        "Proxifier": {
            "log_file": "Proxifier/Proxifier_2k.log",
            "log_format": "\[<Time>\] <Program> - <Content>",
            "regex": [
                r"<\d+\ssec",
                r"([\w-]+\.)+[\w-]+(:\d+)?",
                r"\d{2}:\d{2}(:\d{2})*",
                r"[KGTM]B",
            ],
            "st": 0.6,
            "depth": 3,
            "max": 1000
        },
    }
    
    if selected_datasets:
        lower_keys = {k.lower(): k for k in benchmark_settings.keys()}
        valid_selected = []
        for sd in selected_datasets:
            if sd.lower() not in lower_keys:
                print(f"Erro: Dataset '{sd}' não encontrado no Loghub 2k.")
                print(f"Datasets disponíveis: {', '.join(benchmark_settings.keys())}")
                sys.exit(1)
            valid_selected.append(lower_keys[sd.lower()])
        benchmark_settings = {k: benchmark_settings[k] for k in valid_selected}
        
    file_name = "benchmark_Logscan2k"
    if original_pa:
        file_name += "_original_pa"
    if test_n is not None:
        file_name += "_test"
    file_name += ".csv"
    run_benchmark(input_dir, output_dir, benchmark_settings, result_file=f"benchmark/{file_name}", original_pa=original_pa, test_n=test_n, correct_pa=correct_pa, incorrect_pa=incorrect_pa)

def benchmark_loghub2(original_pa=False, test_n=None, correct_pa=None, incorrect_pa=None, selected_datasets=None):
    if test_n is not None:
        input_dir = "test_dataset/v2/"
        output_dir = "results/loghub2_test/"
    else:
        input_dir = "full_dataset/"
        output_dir = "results/loghub2/"

    # datasets por ordem de tamanho
    benchmark_settings = {
        "Linux": {
            "log_file": "Linux/Linux_full.log",
            "log_format": "<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>",
            "regex": [r"(\d+\.){3}\d+", r"\d{2}:\d{2}:\d{2}"],
            "st": 0.39,
            "depth": 6,
        },
        "Proxifier": {
            "log_file": "Proxifier/Proxifier_full.log",
            "log_format": "\[<Time>\] <Program> - <Content>",
            "regex": [
                r"<\d+\ssec",
                r"([\w-]+\.)+[\w-]+(:\d+)?",
                r"\d{2}:\d{2}(:\d{2})*",
                r"[KGTM]B",
            ],
            "st": 0.6,
            "depth": 3,
            "max": 1000
        },
        "Apache": {
            "log_file": "Apache/Apache_full.log",
            "log_format": "\[<Time>\] \[<Level>\] <Content>",
            "regex": [r"(\d+\.){3}\d+"],
            "st": 0.5,
            "depth": 4,
        },
        "Zookeeper": {
            "log_file": "Zookeeper/Zookeeper_full.log",
            "log_format": "<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>",
            "regex": [r"(/|)(\d+\.){3}\d+(:\d+)?"],
            "st": 0.5,
            "depth": 4,
        },
        "Mac": {
            "log_file": "Mac/Mac_full.log",
            "log_format": "<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>",
            "regex": [r"([\w-]+\.){2,}[\w-]+"],
            "st": 0.7,
            "depth": 6,
        },
        "HealthApp": {
            "log_file": "HealthApp/HealthApp_full.log",
            "log_format": "<Time>\|<Component>\|<Pid>\|<Content>",
            "regex": [],
            "st": 0.2,
            "depth": 4,
        },
        "Hadoop": {
            "log_file": "Hadoop/Hadoop_full.log",
            "log_format": "<Date> <Time> <Level> \[<Process>\] <Component>: <Content>",
            "regex": [r"(\d+\.){3}\d+"],
            "st": 0.5,
            "depth": 4,
        },
        "HPC": {
            "log_file": "HPC/HPC_full.log",
            "log_format": "<LogId> <Node> <Component> <State> <Time> <Flag> <Content>",
            "regex": [r"=\d+"],
            "st": 0.5,
            "depth": 4,
        },
        "OpenStack": {
            "log_file": "OpenStack/OpenStack_full.log",
            "log_format": "<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>",
            "regex": [r"((\d+\.){3}\d+,?)+", r"/.+?\s", r"\d+"],
            "st": 0.5,
            "depth": 5,
        },
        "OpenSSH": {
            "log_file": "OpenSSH/OpenSSH_full.log",
            "log_format": "<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>",
            "regex": [r"(\d+\.){3}\d+", r"([\w-]+\.){2,}[\w-]+"],
            "st": 0.6,
            "depth": 5,
        },
        "BGL": {
            "log_file": "BGL/BGL_full.log",
            "log_format": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
            "regex": [r"core\.\d+"],
            "st": 0.5,
            "depth": 4,
        },
        "HDFS": {
            "log_file": "HDFS/HDFS_full.log",
            "log_format": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
            "regex": [r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"],
            "st": 0.5,
            "depth": 4,
        },
        "Spark": {
            "log_file": "Spark/Spark_full.log",
            "log_format": "<Date> <Time> <Level> <Component>: <Content>",
            "regex": [r"(\d+\.){3}\d+", r"\b[KGTM]?B\b", r"([\w-]+\.){2,}[\w-]+"],
            "st": 0.5,
            "depth": 4,
        },
        "Thunderbird": {
            "log_file": "Thunderbird/Thunderbird_full.log",
            "log_format": "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>",
            "regex": [r"(\d+\.){3}\d+"],
            "st": 0.5,
            "depth": 4,
        },
    }
    
    if selected_datasets:
        lower_keys = {k.lower(): k for k in benchmark_settings.keys()}
        valid_selected = []
        for sd in selected_datasets:
            if sd.lower() not in lower_keys:
                print(f"Erro: Dataset '{sd}' não encontrado no Loghub 2.0.")
                print(f"Datasets disponíveis: {', '.join(benchmark_settings.keys())}")
                sys.exit(1)
            valid_selected.append(lower_keys[sd.lower()])
        benchmark_settings = {k: benchmark_settings[k] for k in valid_selected}
        
    file_name = "benchmark_Logscan"
    if original_pa:
        file_name += "_original_pa"
    if test_n is not None:
        file_name += "_test"
    file_name += ".csv"
    run_benchmark(input_dir, output_dir, benchmark_settings, result_file=f"benchmark/{file_name}", eval_all=True, original_pa=original_pa, test_n=test_n, correct_pa=correct_pa, incorrect_pa=incorrect_pa)

class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f'Erro: {message}\n\n')
        self.print_usage(sys.stderr)
        sys.stderr.write("Use '--help' para ver os parâmetros disponíveis.\n")
        sys.exit(2)

def main():
    print('Logscan')
    print(f"A versão do pacote é: {__version__}")
    
    parser = CustomArgumentParser(
        description="LogScan: Ferramenta automatizada de parsing de logs.\nOs parâmetros --v1, --v2 ou --test são obrigatórios.",
        add_help=False,
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("--help", action="help", default=argparse.SUPPRESS, help="Mostra esta mensagem de ajuda e sai.")
    parser.add_argument("--datasets", type=str, help="Filtra a execução para datasets específicos (ex: --datasets=Linux,Apache).")
    
    version_group = parser.add_mutually_exclusive_group(required=True)
    version_group.add_argument("--v1", action="store_true", help="Run benchmark on Loghub 2k datasets")
    version_group.add_argument("--v2", action="store_true", help="Run benchmark on Loghub 2.0 (full) datasets")
    
    parser.add_argument("--test", nargs='?', const=1, type=int, help="Run on test_dataset with N lines of intermediate steps printed (default 1)")
    parser.add_argument("--original-pa", action="store_true", help="Compute Grouping PA instead of exact matching Parsing Accuracy")
    parser.add_argument("--correct-pa", nargs='?', const=1, type=int, help="Mostrará um output das N primeiras linhas que tiveram o template gerado conforme o gabarito (padrão 1).")
    parser.add_argument("--incorrect-pa", nargs='?', const=1, type=int, help="Mostrará um output das N primeiras linhas que não tiveram o template gerado conforme o gabarito (padrão 1).")
    
    args = parser.parse_args()
    
    selected_datasets = None
    if args.datasets:
        selected_datasets = [d.strip() for d in args.datasets.split(',')]

    if args.v2:
        print("Running Loghub 2.0 Benchmark...")
        benchmark_loghub2(original_pa=args.original_pa, test_n=args.test, correct_pa=args.correct_pa, incorrect_pa=args.incorrect_pa, selected_datasets=selected_datasets)
    elif args.v1:
        print("Running Loghub 2k Benchmark...")
        benchmark(original_pa=args.original_pa, test_n=args.test, correct_pa=args.correct_pa, incorrect_pa=args.incorrect_pa, selected_datasets=selected_datasets)
    elif args.test: # This block is now redundant due to the new --test argument handling
        print("Executando Teste Rápido no Android_2k...")
        android_dataset = pd.read_csv("logs/loghub_2k/Android/Android_2k.log_structured.csv")
        log_scan_android = LogScan(list(android_dataset['Content']), header=False)
        tagger_android, result_dataset = log_scan_android.pipeline()
        result_dataset['EventId'] = android_dataset['EventId']
        result_dataset['EventTemplate'] = android_dataset['EventTemplate']
        if args.original_pa:
            accuracy = original_parsing_accuracy(result_dataset)
        else:
            accuracy = parsing_accuracy(result_dataset)
        print(f"Test Accuracy: {accuracy}")
        result_dataset.to_csv('resultados.csv')

if __name__ == "__main__":
    main()
