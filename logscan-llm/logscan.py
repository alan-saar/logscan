
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

from .auxiliares import is_word, replace_space, has_numbers, word_position, word_counter, remove_repeated, parsing_accuracy
# Funções auxiliares

def log_template(cluster_tagger, log):
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
  def __init__(self, logdata: list, header: bool, header_regex = None):
    print('- Logscan v1.0')
    if header:
      print('-- Header Extraction')
      loglist= [re.sub(f'{header_regex}', '', log) for log in logdata]
      self.data = pd.DataFrame(loglist,columns=['Log'])
    else:
      self.data = pd.DataFrame(logdata,columns=['Log'])

  def clean_data(self):
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
    print('-- TF-IDF Transformer')
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(self.data['CleanLog'])
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    logs_embedding_df = pd.DataFrame(denselist, columns=feature_names)
    return logs_embedding_df

  def dbscanModel(self, logs_embedding_df):
    print('-- DBSCAN')
    clusterModel = DBSCAN(min_samples=2)
    clusterModel.fit(logs_embedding_df)
    self.data['Cluster'] = clusterModel.labels_

  def word_tagger(self):
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
    self.clean_data()
    log_embedding_df = self.tfidf_transformer()
    self.dbscanModel(log_embedding_df)
    tagger = self.word_tagger()
    self.create_templates(tagger)
    return tagger, self.data


def main():
  print('Logscan')
  print(f"The package's version is: {__version__}")

  # Teste
  # android_dataset = pd.read_csv("logs/Andriod_2k.log_structured.csv")
  # log_scan_android = LogScan(list(android_dataset['Content']), header=False)
  # tagger_android, result_dataset = log_scan_android.pipeline()
  # result_dataset['EventId'] = android_dataset['EventId']
  # accuracy = parsing_accuracy(result_dataset)
  # result_dataset.to_csv('resultados.csv')
  benchmark()


def benchmark():
    input_dir = "full_dataset/"  # The input directory of log file
    output_dir = "Logscan-llm_result/"  # The output directory of parsing results
    os.makedirs(output_dir, exist_ok=True)

    benchmark_settings = {
        "HDFS": {
            "log_file": "HDFS/HDFS_full.log",
            "log_format": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
            "regex": [r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"],
            "st": 0.5,
            "depth": 4,
        },
        "Hadoop": {
            "log_file": "Hadoop/Hadoop_full.log",
            "log_format": "<Date> <Time> <Level> \[<Process>\] <Component>: <Content>",
            "regex": [r"(\d+\.){3}\d+"],
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
        "Zookeeper": {
            "log_file": "Zookeeper/Zookeeper_full.log",
            "log_format": "<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>",
            "regex": [r"(/|)(\d+\.){3}\d+(:\d+)?"],
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
        "BGL": {
            "log_file": "BGL/BGL_full.log",
            "log_format": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
            "regex": [r"core\.\d+"],
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
        "Thunderbird": {
            "log_file": "Thunderbird/Thunderbird_full.log",
            "log_format": "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>",
            "regex": [r"(\d+\.){3}\d+"],
            "st": 0.5,
            "depth": 4,
        },
        "Windows": {
            "log_file": "Windows/Windows_full.log",
            "log_format": "<Date> <Time>, <Level>                  <Component>    <Content>",
            "regex": [r"0x.*?\s"],
            "st": 0.7,
            "depth": 5,
        },
        "Linux": {
            "log_file": "Linux/Linux_full.log",
            "log_format": "<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>",
            "regex": [r"(\d+\.){3}\d+", r"\d{2}:\d{2}:\d{2}"],
            "st": 0.39,
            "depth": 6,
        },
        "Mac": {
            "log_file": "Mac/Mac_full.log",
            "log_format": "<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>",
            "regex": [r"([\w-]+\.){2,}[\w-]+"],
            "st": 0.7,
            "depth": 6,
        },
        "Android": {
            "log_file": "Android/Android_full.log",
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
            "log_file": "HealthApp/HealthApp_full.log",
            "log_format": "<Time>\|<Component>\|<Pid>\|<Content>",
            "regex": [],
            "st": 0.2,
            "depth": 4,
        },
        "Apache": {
            "log_file": "Apache/Apache_full.log",
            "log_format": "\[<Time>\] \[<Level>\] <Content>",
            "regex": [r"(\d+\.){3}\d+"],
            "st": 0.5,
            "depth": 4,
        },
        "OpenSSH": {
            "log_file": "OpenSSH/OpenSSH_full.log",
            "log_format": "<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>",
            "regex": [r"(\d+\.){3}\d+", r"([\w-]+\.){2,}[\w-]+"],
            "st": 0.6,
            "depth": 5,
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
        },
    }

    benchmark_result = []
    for dataset, setting in benchmark_settings.items():
        print("\n=== Avaliando: %s ===" % dataset)
        indir = os.path.join(input_dir, os.path.dirname(setting["log_file"]))
        log_file = os.path.basename(setting["log_file"])

        filepath = os.path.join(indir, log_file + "_structured.csv")
        if not os.path.exists(filepath):
            print(f"Skipping {dataset}: {filepath} not found.")
            continue

        test_dataset = pd.read_csv(filepath)
        log_scan = LogScan(list(test_dataset['Content']), header=False)
        tagger, result_dataset = log_scan.pipeline()
        result_dataset['EventId'] = test_dataset['EventId']
        result_dataset.to_csv(os.path.join(output_dir, log_file + "_structured.csv"))

        accuracy = parsing_accuracy(result_dataset)
        benchmark_result.append([dataset, accuracy])


    print("\n=== Resultados ===")
    df_result = pd.DataFrame(benchmark_result, columns=["Dataset", "Accuracy"])
    df_result.set_index("Dataset", inplace=True)
    print(df_result)
    df_result.to_csv("Logscan-llm_benchmark_result.csv", float_format="%.6f")
