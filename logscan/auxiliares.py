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

def parsing_accuracy(data):
  """
  Calculates the parsing accuracy (PA) of the clustering results.

  PA is defined as the ratio of correctly parsed logs to the total number of logs.
  A cluster is correctly parsed if the dominant EventId matches the ground truth.

  Args:
      data (pd.DataFrame): DataFrame containing 'EventId' and 'Cluster' columns.

  Returns:
      float: The parsing accuracy score.
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

  Calculates Parsing Accuracy, Cluster Accuracy, and their combined metric,
  and returns the average of the three along with the individual scores.

  Args:
      data (pd.DataFrame): DataFrame containing 'EventId' and 'Cluster' columns.

  Returns:
      tuple: A tuple containing (average_score, PA, CA, PCA).
  """
  resultado1 = parsing_accuracy(data)
  resultado2 = cluster_accuracy(data)
  resultado3 = parsing_cluster_accuracy(data)
  resultado = (resultado1 + resultado2 + resultado3)/3
  return resultado, resultado1, resultado2, resultado3