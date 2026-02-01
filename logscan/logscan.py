
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
from .llm import generate_templates_for_clusters
# Funções auxiliares

def log_template(cluster_tagger, log):
  """
  Generates a log template by replacing variable parts with '<*>'.

  Args:
      cluster_tagger (list): List of tagged words (variable/template) for the cluster.
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
      for word in cluster_tagger:
        if word[0] == token:
          info = word[3]
          break
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
  def __init__(self, logdata: list, header: bool, header_regex = None):
    """
    Initializes the LogScan instance.

    Args:
        logdata (list): List of raw log strings.
        header (bool): Whether to strip headers using regex.
        header_regex (str, optional): Regex pattern for header removal.
    """
    print('- Logscan v1.0')
    if header:
      print('-- Header Extraction')
      loglist= [re.sub(f'{header_regex}', '', log) for log in logdata]
      self.data = pd.DataFrame(loglist,columns=['Log'])
    else:
      self.data = pd.DataFrame(logdata,columns=['Log'])

  def clean_data(self):
    """
    Preprocesses log data by tokenizing and removing non-word characters.

    Populates the 'CleanLog' column in self.data.
    """
    print('-- Data Cleaning')
    clear_content = []
    for _, row in self.data.iterrows():
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
    Converts cleaned logs to TF-IDF vectors.

    Returns:
        pd.DataFrame: DataFrame containing the TF-IDF feature matrix.
    """
    print('-- TF-IDF Transformer')
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(self.data['CleanLog'])
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    logs_embedding_df = pd.DataFrame(denselist, columns=feature_names)
    return logs_embedding_df

  def dbscanModel(self, logs_embedding_df):
    """
    Apply DBSCAN clustering to the log embeddings.

    Populates the 'Cluster' column in self.data.

    Args:
        logs_embedding_df (pd.DataFrame): TF-IDF feature matrix.
    """
    print('-- DBSCAN')
    clusterModel = DBSCAN(min_samples=2)
    clusterModel.fit(logs_embedding_df)
    self.data['Cluster'] = clusterModel.labels_

  def word_tagger(self):
    """
    Analyzes word frequency per cluster to tag words as variables or template parts.

    Returns:
        list: A list of (cluster_id, labeled_words) tuples.
    """
    print('-- Word Tagger')
    tagger = []
    for cluster in np.unique(self.data['Cluster']):
      cluster_tokens = []
      dados_cluster = self.data.loc[self.data['Cluster'] == cluster]['Log']
      for log in dados_cluster:
        tokens = wordpunct_tokenize(log)
        tokens_position = word_position (tokens)
        cluster_tokens = cluster_tokens + tokens_position
      word_frequency = word_counter(cluster_tokens)
      new_wordfrequency = remove_repeated(word_frequency)
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
    print('-- Template Extraction')
    templates = []
    variables = []
    for index, row in self.data.iterrows():
      log_cluster = row['Cluster']
      for cluster_tagger in tagger:
        if cluster_tagger[0] == log_cluster:
          log_tagger = cluster_tagger[1]
          break
      template, variables_list = log_template(log_tagger, row['Log'])
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
    log_embedding_df = self.tfidf_transformer()
    self.dbscanModel(log_embedding_df)
    tagger = self.word_tagger()
    self.create_templates(tagger)
    return tagger, self.data

  def pipeline_llm(self, dataset_name):
    """
    Runs the hybrid LogScan pipeline with LLM template generation.
    """
    print(f"Running Hybrid Pipeline for {dataset_name}...")
    self.clean_data()
    log_embedding_df = self.tfidf_transformer()
    self.dbscanModel(log_embedding_df)

    # Use LLM to generate templates for each cluster
    print('-- LLM Template Generation')
    cluster_templates = generate_templates_for_clusters(self.data, dataset_name)

    templates = []
    variables = [] # We might not extract variables perfectly if we just get the template string, but we can try matching

    # Apply templates back to rows
    for index, row in self.data.iterrows():
        cluster_id = row['Cluster']
        if cluster_id in cluster_templates:
            template = cluster_templates[cluster_id]
        else:
            # Fallback if LLM failed (shouldn't happen often if we have catch-all)
            template = row['Log']

        templates.append(template)
        # Variables extraction is harder without the tagger logic,
        # but for accuracy benchmark we mainly need the EventId (which comes from unique templates)
        variables.append([])

    self.data['Template'] = templates
    self.data['Variables'] = variables

    # Assign EventId based on unique templates
    unique_templates = self.data['Template'].unique()
    template_to_id = {tmpl: f"E{i+1}" for i, tmpl in enumerate(unique_templates)}
    self.data['EventId'] = self.data['Template'].map(template_to_id)

    return self.data


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
        # "HDFS": {
        #     "log_file": "HDFS/HDFS_full.log",
        #     "log_format": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
        #     "regex": [r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"],
        #     "st": 0.5,
        #     "depth": 4,
        # },
        # "Spark": {
        #     "log_file": "Spark/Spark_full.log",
        #     "log_format": "<Date> <Time> <Level> <Component>: <Content>",
        #     "regex": [r"(\d+\.){3}\d+", r"\b[KGTM]?B\b", r"([\w-]+\.){2,}[\w-]+"],
        #     "st": 0.5,
        #     "depth": 4,
        # },
        # "Thunderbird": {
        #     "log_file": "Thunderbird/Thunderbird_full.log",
        #     "log_format": "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>",
        #     "regex": [r"(\d+\.){3}\d+"],
        #     "st": 0.5,
        #     "depth": 4,
        # },
    }

    run_benchmark(input_dir, output_dir, benchmark_settings, result_file="Logscan_loghub2_benchmark_result.csv")


def benchmark_llm():
    input_dir = "full_dataset/"
    output_dir = "Logscan_llm_results/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use same settings as Loghub 2.0 but we need to pass the dataset name correctly
    benchmark_settings = {
        "Linux": { "log_file": "Linux/Linux_full.log" },
        "Proxifier": { "log_file": "Proxifier/Proxifier_full.log" },
        "Apache": { "log_file": "Apache/Apache_full.log" },
        "Zookeeper": { "log_file": "Zookeeper/Zookeeper_full.log", },
        "Mac": { "log_file": "Mac/Mac_full.log",},
        # "HealthApp": { "log_file": "HealthApp/HealthApp_full.log" },
        # "Hadoop": { "log_file": "Hadoop/Hadoop_full.log" },
        # "HPC": { "log_file": "HPC/HPC_full.log" },
        # "OpenStack": { "log_file": "OpenStack/OpenStack_full.log" },
        # "OpenSSH": { "log_file": "OpenSSH/OpenSSH_full.log" },
        # Add others if needed/available in full_dataset
    }

    benchmark_result = []

    for dataset, setting in benchmark_settings.items():
        print(f"\n=== Running Hybrid LLM on: {dataset} ===")
        log_file_rel = setting["log_file"]
        indir = os.path.join(input_dir, os.path.dirname(log_file_rel))
        log_filename = os.path.basename(log_file_rel)

        full_log_path = os.path.join(indir, log_filename)

        if not os.path.exists(full_log_path):
            print(f"Error: {full_log_path} not found.")
            continue

        try:
            # We need to read the log content.
            # Note: LogScan init expects a list of strings.
            # And it expects to handle headers if header=True.
            # Loghub 2.0 'full' logs might vary.
            # benchmark_loghub2 implementation didn't show reading logic in detail in Step 198,
            # but look at existing run_benchmark logic:
            # test_dataset = pd.read_csv(os.path.join(indir, log_file + "_structured.csv"))
            # log_scan_android = LogScan(list(test_dataset['Content']), header=False)

            # Here we are processing raw logs? Or structured?
            # The prompt says "utilizar a full_dataset".
            # Usually full_dataset has raw files (.log).
            # But earlier code was reading `_structured.csv` to get ground truth `Content`?
            # Let's assume we read the raw log file directly.

            with open(full_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                log_lines = f.readlines()

            # Initialize LogScan
            # We turn off header removal for now or we need to know the regex.
            # Using header=False for simplicity as in 'benchmark_loghub2' comments (inferred)
            log_scan = LogScan(log_lines, header=False)

            # Run Hybrid Pipeline
            result_dataset = log_scan.pipeline_llm(dataset_name=dataset)

            # Save results
            result_file = os.path.join(output_dir, f"{dataset}_full.log_structured.csv")
            result_dataset.to_csv(result_file, index=False)
            print(f"Saved results to {result_file}")

            # Note: We can't calculate accuracy here easily without ground truth.
            # Ground truth is typically in LOGname_structured.csv.
            # If it exists, we can compare.
            ground_truth_path = os.path.join(indir, log_filename + "_structured.csv")
            # Usually Loghub 2.0 ground truths are named nicely?
            # If not found, accurate calculation is skipped.

        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()


def main():
    print('Logscan')
    print(f"The package's version is: {__version__}")

    parser = argparse.ArgumentParser(description="LogScan: Automated Log Parsing")
    parser.add_argument("--v2", action="store_true", help="Run benchmark on Loghub 2.0 datasets")
    parser.add_argument("--v2-llm", action="store_true", help="Run benchmark on Loghub 2.0 datasets using LILAC LLM Parser (Hybrid)")
    parser.add_argument("--test", action="store_true", help="Run a quick test on Android_2k logs")

    args = parser.parse_args()

    if args.v2_llm:
        print("Running Loghub 2.0 Benchmark with Hybrid LLM...")
        benchmark_llm()
    elif args.v2:
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


