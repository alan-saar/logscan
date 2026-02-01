import openai
import os
import regex as re
import time
import string
import json
import random
import textdistance
import pandas as pd
import numpy as np
from collections import defaultdict

# ==========================================
# Utils & Configuration
# ==========================================

def get_keys_from_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            api_base = file.readline().strip()
            key_str = file.readline().strip()
        return api_base, key_str
    return None, None

def setup_openai():
    # Try looking for openai_key.txt in project root
    # We assume this file is in logscan/logscan/llm.py, so project root is ../../
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))
    key_path = os.path.join(project_root, 'openai_key.txt')
    
    api_base, api_key = get_keys_from_file(key_path)
    
    if not api_key and "OPENAI_KEY" in os.environ:
        api_base = "https://api.openai.com/v1"
        api_key = os.environ["OPENAI_KEY"]
    
    if api_key:
        openai.api_base = api_base
        openai.api_key = api_key
        return True
    else:
        print("Warning: OpenAI Key not found in openai_key.txt or environment variables.")
        return False

# ==========================================
# Parsing Cache (Partial Port for Validation)
# ==========================================

class ParsingCache(object):
    def __init__(self):
        self.template_tree = {}
        self.template_list = []
    
    def add_templates(self, event_template):
        template_tokens = self.message_split(event_template)
        if not template_tokens or event_template == "<*>":
            return -1
        # Simplified insertion for validation checks
        return 1

    def match_event(self, log):
        # Simplified match for validation
        return self.tree_match(self.template_tree, log)

    def message_split(self, message):
        punc = "!\"#$%&'()+,-/:;=?@.[\]^_`{|}~"
        splitters = "\s\\" + "\\".join(punc)
        splitter_regex = re.compile("([{}])".format(splitters))
        tokens = re.split(splitter_regex, message)
        tokens = list(filter(lambda x: x != "", tokens))
        # Post process tokens logic simplified
        tokens = [token.strip() for token in tokens if token != "" and token != ' ']
        return tokens

    def tree_match(self, match_tree, log_content):
        # We only need this to check if "NoMatch" is returned to validate if the template matches the log
        # For the purpose of 'query_template_from_gpt_with_check', we want to know if the generated template
        # matches the log it was generated from. This implies we need a matcher.
        
        # Since implementing the full tree match here is complex, and we are using this for specific 
        # validation of "does X match Y", we can use a simpler regex approach for checking.
        return ("Match", 1, "") 

# Note: The original ParsingCache is complex. 
# For the hybrid approach where we generate a template for a cluster, 
# we mainly trust the LLM or doing a simple regex check is enough.
# I will implement a simpler regex-based check function.

def check_template_matches_log(template, log):
    """
    Checks if a generated template matches the log line.
    Template has <*>. We convert it to regex matching anything.
    """
    if template == "": return False
    
    # Escape special regex chars in template, except <*>
    # We replace <*> with a placeholder first to avoid escaping, then convert to .*?
    temp_placeholder = "___VAR___"
    safe_template = template.replace("<*>", temp_placeholder)
    safe_template = re.escape(safe_template)
    regex_pattern = safe_template.replace(temp_placeholder, r"(.*?)")
    regex_pattern = "^" + regex_pattern + "$"
    
    return bool(re.match(regex_pattern, log))

# ==========================================
# Post Processing
# ==========================================

def correct_single_template(template):
    template = template.strip()
    template = re.sub(r'\s+', ' ', template)
    
    # Apply standard cleaning rules
    token_delimiters = [r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&']
    tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)
    new_tokens = []
    for token in tokens:
        if re.match(r'^\d+$', token):
            token = '<*>'
        new_tokens.append(token)
    template = ''.join(new_tokens)

    # Merge consecutive variables
    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        template = re.sub(r'<\*><\*>', '<*>', template)
        if prev == template:
            break
            
    # Clean up formatting
    replacements = [
        (" #<*># ", " <*> "), (" #<*> ", " <*> "), ("<*>:<*>", "<*>"),
        ("<*>#<*>", "<*>"), ("<*>/<*>", "<*>"), ("<*>@<*>", "<*>"),
        ("<*>.<*>", "<*>"), (' "<*>" ', ' <*> '), (" '<*>' ", " <*> "),
        ("<*><*>", "<*>")
    ]
    for old, new in replacements:
        while old in template:
            template = template.replace(old, new)
            
    return template

def post_process_template(template, regs_common=[]):
    pattern = r'\{(\w+)\}'
    template = re.sub(pattern, "<*>", template)
    for reg in regs_common:
        template = reg.sub("<*>", template)
    template = correct_single_template(template)
    
    static_part = template.replace("<*>", "")
    punc = string.punctuation
    for s in static_part:
        if s != ' ' and s not in punc:
            return template, True
            
    # "Get a too general template. Error."
    return "", False

# ==========================================
# Prompt Selection
# ==========================================

def jaccard_distance(x, y):
    x_set = set(x.split())
    y_set = set(y.split())
    return 1 - (len(x_set.intersection(y_set)) / len(x_set.union(y_set)) if x_set.union(y_set) else 0)

def clean(s):
    # Simplified cleaning
    s = re.sub(r'(\d+\.){3}\d+(:\d+)?', " ", s)
    s = re.sub(r'(:|\(|\)|=|,|"|\{|\}|@|$|\[|\]|\||;|\.)', ' ', s)
    return " ".join(s.lower().split())

def prompt_select(prompts, log, demonstration, selection_method="LILAC"):
    if demonstration == 0 or not prompts:
        return []
    
    examples = []
    if selection_method == "random":
        result = random.choices(prompts, k=demonstration)
    else:
        # LILAC selection (Jaccard similarity)
        L = prompts
        log_clean = clean(log)
        # Calculate similarity (re-using the logic from source)
        # Note: The source code used distance, then sorted. 
        # But Jaccard distance: 0=identical, 1=different.
        # We want MOST similar, so lowest distance.
        scores = []
        for d in L:
            dist = jaccard_distance(clean(d['query']), log_clean)
            scores.append((dist, d))
        
        # Sort by distance (ascending)
        scores.sort(key=lambda x: x[0])
        result = [x[1] for x in scores[:demonstration]]
        
    for x in result:
        examples.append({'query': f"Log message: `{x['query']}`",
                         'answer': f"Log template: `{x['answer']}`"})
    return examples

# ==========================================
# GPT & LLM Interaction
# ==========================================

def infer_llm(instruction, exemplars, query, log_message, model='gpt-3.5-turbo-0613', temperature=0.0):
    messages = [{"role": "system", "content": "You are an expert of log parsing, and now you will help to do log parsing."},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": "Sure, I can help you with log parsing."}]

    if exemplars:
        for exemplar in exemplars:
            messages.append({"role": "user", "content": exemplar['query']})
            messages.append({"role": "assistant", "content": exemplar['answer']})
            
    messages.append({"role": "user", "content": query})
    
    retry_times = 0
    while retry_times < 3:
        try:
            answers = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False
            )
            return [response["message"]["content"] for response in answers["choices"] if response['finish_reason'] != 'length'][0]
        except Exception as e:
            print(f"GPT Error: {e}")
            retry_times += 1
            time.sleep(1)
            
    # Fallback to identify match
    if exemplars and len(exemplars) > 0:
        # Simplified recursive retry fallback
        pass
        
    return 'Log message: `{}`'.format(log_message)

def get_response_from_openai(query, examples=[], model='gpt-3.5-turbo-0613'):
    instruction = "I want you to act like an expert of log parsing. I will give you a log message delimited by backticks. You must identify and abstract all the dynamic variables in logs with {placeholder} and output a static log template. Print the input log's template delimited by backticks."
    
    # Default example if none
    if not examples:
        examples = [{'query': 'Log message: `try to connected to host: 172.16.254.1, finished.`', 
                     'answer': 'Log template: `try to connected to host: {ip_address}, finished.`'}]
                     
    question = 'Log message: `{}`'.format(query)
    return infer_llm(instruction, examples, question, query, model)

def query_template(log_message, examples=[], model='gpt-3.5-turbo-0613'):
    if len(log_message.split()) <= 1:
        return log_message, False

    response = get_response_from_openai(log_message, examples, model)
    
    # Parse response to find template
    lines = response.split('\n')
    log_template = None
    for line in lines:
        if "Log template:" in line:
            log_template = line
            break
    if not log_template:
        for line in lines:
            if "`" in line:
                log_template = line
                break
                
    if log_template:
        start = log_template.find('`') + 1
        end = log_template.rfind('`')
        if start > 0 and end > start:
            return log_template[start:end], True
            
    return log_template if log_template else log_message, False

def generate_template_with_check(log_message, dataset_examples=[], model="gpt-3.5-turbo-0613"):
    template, flag = query_template(log_message, dataset_examples, model)
    
    if len(template) == 0 or not flag:
        return post_process_template(log_message)[0]
    
    clean_template, valid = post_process_template(template)
    if valid:
        # Validation check: does the template match the log?
        if check_template_matches_log(clean_template, log_message):
            return clean_template
        else:
            print(f"LLM Error: Template '{clean_template}' does not match log '{log_message}'.")
            
    return post_process_template(log_message)[0]

# ==========================================
# Main Interface for Logscan
# ==========================================

def load_prompt_cases(dataset_name, shot=8):
    # Find full_dataset path from project root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))
    
    json_path = os.path.join(project_root, "full_dataset", "sampled_examples", dataset_name, f"{shot}shot.json")
    
    cases = []
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            for line in f:
                cases.append(json.loads(line))
    return cases

def generate_templates_for_clusters(df, dataset_name, shot=8, model="gpt-3.5-turbo-0125"):
    """
    Generates templates for each cluster in the dataframe using LLM.
    
    Args:
        df: DataFrame with 'Cluster' and 'Log' columns.
        dataset_name: Name of the dataset (e.g., 'HealthApp') for few-shot loading.
        
    Returns:
        dict: {cluster_id: template_string}
    """
    if not setup_openai():
        print("Skipping LLM generation as OpenAI key is missing.")
        return {}

    prompt_cases = load_prompt_cases(dataset_name, shot)
    print(f"Loaded {len(prompt_cases)} few-shot examples for {dataset_name}.")
    
    cluster_templates = {}
    clusters = np.unique(df['Cluster'])
    
    print(f"Generating templates for {len(clusters)} clusters...")
    
    for cluster_id in clusters:
        # Get one representative log from the cluster (first one)
        cluster_logs = df[df['Cluster'] == cluster_id]['Log'].tolist()
        if not cluster_logs:
            continue
            
        representative_log = cluster_logs[0]
        
        # Select examples
        examples = prompt_select(prompt_cases, representative_log, shot)
        
        # Query LLM
        print(f"Cluster {cluster_id}: Querying LLM...")
        template = generate_template_with_check(representative_log, examples, model)
        print(f"  -> Template: {template}")
        
        cluster_templates[cluster_id] = template
        
    return cluster_templates
