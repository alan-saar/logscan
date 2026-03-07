
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
  def _update_progress(self, percent):
    bars = int(percent / 5)
    if bars > 20: bars = 20
    if percent > 100: percent = 100
    bar_str = '#' * bars + ' ' * (20 - bars)
    sys.stdout.write(f'\r[{bar_str}] {percent}%')
    sys.stdout.flush()

  def __init__(self, logdata: list, header: bool, header_regex = None):
    """
    Initializes the LogScan instance.

    Args:
        logdata (list): List of raw log strings.
        header (bool): Whether to strip headers using regex.
        header_regex (str, optional): Regex pattern for header removal.
    """
    # print('- Logscan v1.0')
    self._update_progress(0)
    if header:
      # print('-- Header Extraction')
      loglist= [re.sub(f'{header_regex}', '', log) for log in logdata]
      self.data = pd.DataFrame(loglist,columns=['Log'])
    else:
      self.data = pd.DataFrame(logdata,columns=['Log'])

  def clean_data(self):
    """
    Preprocesses log data by tokenizing and removing non-word characters.

    Populates the 'CleanLog' column in self.data.
    """
    # print('-- Data Cleaning')
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

  def tfidf_transformer(self):
    """
    Converts unique cleaned logs to TF-IDF vectors.

    Returns:
        tuple: (vectors, unique_clean_logs)
    """
    # print('-- TF-IDF Transformer')
    self._update_progress(20)
    unique_clean_logs = self.data['CleanLog'].drop_duplicates().reset_index(drop=True)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(unique_clean_logs)
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
    self._update_progress(30)
    clusterModel = DBSCAN(min_samples=2)
    clusterModel.fit(vectors)
    
    # Map clusters back to all rows based on their CleanLog
    cluster_map = dict(zip(unique_clean_logs, clusterModel.labels_))
    self.data['Cluster'] = self.data['CleanLog'].map(cluster_map)

  def word_tagger(self):
    """
    Analyzes word frequency per cluster to tag words as variables or template parts.

    Returns:
        list: A list of (cluster_id, labeled_words) tuples.
    """
    # print('-- Word Tagger')
    self._update_progress(40)
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
    return tagger

  def create_templates(self, tagger):
    """
    Generates templates for all logs based on the tagger results.

    Populates 'Template' and 'Variables' columns in self.data.

    Args:
        tagger (list): Output from word_tagger().
    """
    # print('-- Template Extraction')
    self._update_progress(70)
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
    self.clean_data()
    vectors, unique_clean_logs = self.tfidf_transformer()
    self.dbscanModel(vectors, unique_clean_logs)
    tagger = self.word_tagger()
    self.create_templates(tagger)
    self._update_progress(100)
    print()
    return tagger, self.data


# Main logic handled below



def parsing_accuracy(data):
    log_per_template =  data['EventId'].value_counts().to_dict()
    correct = 0
    for cluster in np.unique(data['Cluster']):
        data_cluster = data.loc[data['Cluster'] == cluster]
        log_per_template_cluster =  data_cluster['EventId'].value_counts().to_dict()
        for eventid in np.unique(data_cluster['EventId']):
            if log_per_template[eventid] == log_per_template_cluster[eventid]:
                correct = correct + log_per_template_cluster[eventid]
    return correct/len(data)

import argparse

def run_benchmark(input_dir, output_dir, settings, result_file="Logscan_benchmark_result.csv"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    benchmark_result = []
    for dataset, setting in settings.items():
        print("\n=== Avaliando: %s ===" % dataset)
        indir = os.path.join(input_dir, os.path.dirname(setting["log_file"]))
        log_file = os.path.basename(setting["log_file"])

        try:
            test_dataset = pd.read_csv(os.path.join(indir, log_file + "_structured.csv"))
            log_scan_android = LogScan(list(test_dataset['Content']), header=False)
            tagger_android, result_dataset = log_scan_android.pipeline()
            result_dataset['EventId'] = test_dataset['EventId']
            result_dataset.to_csv(os.path.join(output_dir, log_file + "_structured.csv"))

            accuracy = parsing_accuracy(result_dataset)
            benchmark_result.append([dataset, accuracy])
        except Exception as e:
            print(f"Error processing {dataset}: {e}")

    print("\n=== Resultados ===")
    df_result = pd.DataFrame(benchmark_result, columns=["Dataset", "Accuracy"])
    df_result.set_index("Dataset", inplace=True)
    print(df_result)
    df_result.to_csv(result_file, float_format="%.6f")

def benchmark():
    input_dir = "logs/loghub_2k/"
    output_dir = "Logscan_result/"
    
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
    
    run_benchmark(input_dir, output_dir, benchmark_settings)

def benchmark_loghub2():
    input_dir = "full_dataset/"
    output_dir = "Logscan_loghub2_results/"

    # datasets por ordem de tamanho
    benchmark_settings = {
        # "Linux": {
        #     "log_file": "Linux/Linux_full.log",
        #     "log_format": "<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>",
        #     "regex": [r"(\d+\.){3}\d+", r"\d{2}:\d{2}:\d{2}"],
        #     "st": 0.39,
        #     "depth": 6,
        # },
        # "Proxifier": {
        #     "log_file": "Proxifier/Proxifier_full.log",
        #     "log_format": "\[<Time>\] <Program> - <Content>",
        #     "regex": [
        #         r"<\d+\ssec",
        #         r"([\w-]+\.)+[\w-]+(:\d+)?",
        #         r"\d{2}:\d{2}(:\d{2})*",
        #         r"[KGTM]B",
        #     ],
        #     "st": 0.6,
        #     "depth": 3,
        #     "max": 1000
        # },
        # "Apache": {
        #     "log_file": "Apache/Apache_full.log",
        #     "log_format": "\[<Time>\] \[<Level>\] <Content>",
        #     "regex": [r"(\d+\.){3}\d+"],
        #     "st": 0.5,
        #     "depth": 4,
        # },
        # "Zookeeper": {
        #     "log_file": "Zookeeper/Zookeeper_full.log",
        #     "log_format": "<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>",
        #     "regex": [r"(/|)(\d+\.){3}\d+(:\d+)?"],
        #     "st": 0.5,
        #     "depth": 4,
        # },
        # "Mac": {
        #     "log_file": "Mac/Mac_full.log",
        #     "log_format": "<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>",
        #     "regex": [r"([\w-]+\.){2,}[\w-]+"],
        #     "st": 0.7,
        #     "depth": 6,
        # },
        # "HealthApp": {
        #     "log_file": "HealthApp/HealthApp_full.log",
        #     "log_format": "<Time>\|<Component>\|<Pid>\|<Content>",
        #     "regex": [],
        #     "st": 0.2,
        #     "depth": 4,
        # },
        # "Hadoop": {
        #     "log_file": "Hadoop/Hadoop_full.log",
        #     "log_format": "<Date> <Time> <Level> \[<Process>\] <Component>: <Content>",
        #     "regex": [r"(\d+\.){3}\d+"],
        #     "st": 0.5,
        #     "depth": 4,
        # },
        # "HPC": {
        #     "log_file": "HPC/HPC_full.log",
        #     "log_format": "<LogId> <Node> <Component> <State> <Time> <Flag> <Content>",
        #     "regex": [r"=\d+"],
        #     "st": 0.5,
        #     "depth": 4,
        # },
        # "OpenStack": {
        #     "log_file": "OpenStack/OpenStack_full.log",
        #     "log_format": "<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>",
        #     "regex": [r"((\d+\.){3}\d+,?)+", r"/.+?\s", r"\d+"],
        #     "st": 0.5,
        #     "depth": 5,
        # },
        # "OpenSSH": {
        #     "log_file": "OpenSSH/OpenSSH_full.log",
        #     "log_format": "<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>",
        #     "regex": [r"(\d+\.){3}\d+", r"([\w-]+\.){2,}[\w-]+"],
        #     "st": 0.6,
        #     "depth": 5,
        # },
        # "BGL": {
        #     "log_file": "BGL/BGL_full.log",
        #     "log_format": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
        #     "regex": [r"core\.\d+"],
        #     "st": 0.5,
        #     "depth": 4,
        # },
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
    
    run_benchmark(input_dir, output_dir, benchmark_settings, result_file="Logscan_loghub2_benchmark_result.csv")


def main():
    print('Logscan')
    print(f"The package's version is: {__version__}")
    
    parser = argparse.ArgumentParser(description="LogScan: Automated Log Parsing")
    parser.add_argument("--v2", action="store_true", help="Run benchmark on Loghub 2.0 datasets")
    parser.add_argument("--test", action="store_true", help="Run a quick test on Android_2k logs")
    
    args = parser.parse_args()

    if args.v2:
        print("Running Loghub 2.0 Benchmark...")
        benchmark_loghub2()
    elif args.test:
        print("Running Test on Android_2k...")
        android_dataset = pd.read_csv("logs/Android_2k.log_structured.csv")
        log_scan_android = LogScan(list(android_dataset['Content']), header=False)
        tagger_android, result_dataset = log_scan_android.pipeline()
        result_dataset['EventId'] = android_dataset['EventId']
        accuracy = parsing_accuracy(result_dataset)
        print(f"Test Accuracy: {accuracy}")
        result_dataset.to_csv('resultados.csv')
    else:
        # Default behavior: run original benchmark
        print("Running Standard Loghub 2k Benchmark...")
        benchmark()

if __name__ == "__main__":
    main()


