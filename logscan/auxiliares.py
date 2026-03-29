import re
import numpy as np

"""
Auxiliary functions for the LogScan algorithm.

This module contains helper functions for string processing,
word frequency analysis, and accuracy evaluation of the clustering results.
"""

def is_word(inputString):
  """
  Checks if a string consists only of alphabetic characters.

  Args:
      inputString (str): The string to check.

  Returns:
      bool: True if the string contains only letters, False otherwise.
  """
  return bool(re.search(r'^[a-zA-Z]+$', inputString))

def replace_space(old_string):
  """
  Replaces spaces in a string with a special placeholder tag.

  Args:
      old_string (str): The original string.

  Returns:
      str: The string with spaces replaced by '_IS_SPACE_'.
  """
  new_string = old_string.replace(" ", " _IS_SPACE_ ")
  return new_string

def has_numbers(inputString):
  """
  Checks if a string contains any numeric digits.

  Args:
      inputString (str): The string to check.

  Returns:
      bool: True if the string contains digits, False otherwise.
  """
  return bool(re.search(r'\d', inputString))

def word_position (wordlist):
  """
  Annotates a list of words with their position indices.

  Args:
      wordlist (list): A list of word tokens.

  Returns:
      list: A list of tuples, where each tuple contains (word, position).
  """
  word_position_list = []
  position = 0
  for word in wordlist:
    word_position_list.append((word, position))
    position = position + 1
  return word_position_list

def word_counter(wordlist):
  """
  Counts the frequency of each word (tuple) in the list.

  Args:
      wordlist (list): A list of word tuples/tokens.

  Returns:
      list: A list of tuples (word, position, frequency).
  """
  wordfreq = []
  listw = wordlist.copy()
  word_frequency = []
  for w in listw:
    frequency = listw.count(w)
    wordfreq.append(frequency)
    word_frequency.append((w[0], w[1], frequency))
  return word_frequency

def remove_repeated(wordfrequency):
  """
  Removes duplicate entries from the word frequency list.

  Args:
      wordfrequency (list): The list of word frequency tuples.

  Returns:
      list: A list of unique word frequency tuples.
  """
  new_wordfrequency = []
  for word in wordfrequency:
    if word not in new_wordfrequency:
      new_wordfrequency.append(word)
  return new_wordfrequency

# Funcoes de avaliacao

def original_parsing_accuracy(data):
  """
  Calculates Grouping parsing accuracy (Original PA) of the clustering results.

  This is a Group Accuracy metric calculating the ratio of correctly grouped logs 
  to the total number of logs based on EventId.
  """
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
  """
  Calculates the exact sequence Parsing Accuracy (PA) of the clustering results.

  PA is defined as the proportion of correctly parsed log messages to the total number 
  of log messages. A log message is regarded as correctly parsed if, and only if, all tokens 
  of templates and variables are accurately identified. This requires both EventTemplate and
  Template columns to exist in data.

  Args:
      data (pd.DataFrame): DataFrame containing 'EventTemplate' and 'Template' columns.

  Returns:
      float: The exact match parsing accuracy score.
  """
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

def cluster_accuracy(data):
  """
  Calculates the cluster accuracy (CA) of the clustering results.

  CA measures the precision of grouping logs of the same template into the same cluster.
  It counts a cluster as correct only if it contains logs from a single EventId.

  Args:
      data (pd.DataFrame): DataFrame containing 'EventId' and 'Cluster' columns.

  Returns:
      float: The cluster accuracy score.
  """
  # CA - 
  correct = 0
  for cluster in np.unique(data['Cluster']):
    data_cluster = data.loc[data['Cluster'] == cluster]
    log_per_template_cluster =  data_cluster['EventId'].value_counts().to_dict()
    for eventid in np.unique(data_cluster['EventId']):
      if len(np.unique(data_cluster['EventId'])) == 1:
        correct = correct + log_per_template_cluster[eventid]
  return correct/len(data)

def parsing_cluster_accuracy(data):
  """
  Calculates the combined Parsing and Cluster Accuracy.

  This metric considers a match only if the cluster corresponds to a single EventId
  and matches the most frequent EventId distribution.

  Args:
      data (pd.DataFrame): DataFrame containing 'EventId' and 'Cluster' columns.

  Returns:
      float: The combined accuracy score.
  """
  # Parsing Accuracy + Cluster Accuracy
  log_per_template =  data['EventId'].value_counts().to_dict()
  correct = 0
  for cluster in np.unique(data['Cluster']):
    data_cluster = data.loc[data['Cluster'] == cluster]
    log_per_template_cluster =  data_cluster['EventId'].value_counts().to_dict()
    for eventid in np.unique(data_cluster['EventId']):
      if log_per_template[eventid] == log_per_template_cluster[eventid] and len(np.unique(data_cluster['EventId'])) == 1:
        correct = correct + log_per_template_cluster[eventid]
  return correct/len(data)

def cluster_evaluation(data):
  """
  Computes a comprehensive evaluation of the clustering results.

  Calculates Original Parsing Accuracy, Cluster Accuracy, and their combined metric,
  and returns the average of the three along with the individual scores.

  Args:
      data (pd.DataFrame): DataFrame containing 'EventId' and 'Cluster' columns.

  Returns:
      tuple: A tuple containing (average_score, PA, CA, PCA).
  """
  resultado1 = original_parsing_accuracy(data)
  resultado2 = cluster_accuracy(data)
  resultado3 = parsing_cluster_accuracy(data)
  resultado = (resultado1 + resultado2 + resultado3)/3
  return resultado, resultado1, resultado2, resultado3

def template_accuracy(data):
  """
  Calculates Template Accuracy metrics.

  Template Accuracy strictly checks if a parsed template exactly matches 
  a template existing in the ground truth EventTemplates.

  Args:
      data (pd.DataFrame): DataFrame containing 'EventTemplate' and 'Template' columns.

  Returns:
      tuple: A tuple containing (FTA, PTA, RTA).
  """
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