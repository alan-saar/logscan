
"""
LogScan LLM Module.

This module implements the LogScan LLM funcionality for algorithm for automated log parsing and template extraction.
"""

import openai
import os

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

def test_openai_key():
    setup_openai()

    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo-0613", # deprecated
        model="gpt-3.5-turbo-0125", # deprecated
        messages=[
            {"role": "user", "content": "Teste rápido: diga OK."}
        ]
    )

    print(response["choices"][0]["message"]["content"])

