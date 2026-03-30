import pandas as pd
import numpy as np
import re
from logscan.llm import call_openai_api, get_llm_calls

def run_llm_refinement(logscan_instance, debug_llm=None):
    """
    Executes the LLM1 approach: Post-processing refinement of extracted templates.
    """
    dataset = logscan_instance.data
    clusters = np.unique(dataset['Cluster'])
    
    refined_templates_map = {}
    
    # debug_llm is either None (disabled), -1 (all calls), or N (first N calls)
    print("\nIniciando refinamento por LLM nas templates extraídas...")
    for i, cluster in enumerate(clusters):
        # Print progress
        print(f"\rProcessando cluster {i+1}/{len(clusters)} com LLM...", end="", flush=True)

        if cluster == -1:
            continue

        cluster_logs = dataset.loc[dataset['Cluster'] == cluster]['Log'].unique()
        sample_logs = cluster_logs[:5]
        
        system_prompt = (
            "I want you to act like an expert in log parsing. Your task is to identify all the "
            "dynamic variables in logs, replace them with <*>, and output a static log template. "
            "Please print the final log's template wrapped by backticks. If multiple similar logs are provided, "
            "extract the single template that matches all of them."
        )
        
        user_content = "\n".join([f"`{log}`" for log in sample_logs])
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        try:
            response = call_openai_api(messages)
            match = re.search(r'`+(.*?)`+', response, re.DOTALL)
            if match:
                refined_template = match.group(1).strip()
            else:
                refined_template = response.strip()
                
            calls_made = get_llm_calls()
            if debug_llm is not None:
                if debug_llm < 0 or calls_made <= debug_llm:
                    print(f"\n\n[DEBUG LLM CALL {calls_made} - Cluster {cluster}]")
                    print(f"--- Prompt Content ---\n{user_content}")
                    print(f"--- Resposta ---\n{response}")
                    print(f"--- Extrato Refinado ---\n{refined_template}")
                    print("-" * 40)
                    
            refined_templates_map[cluster] = refined_template
            
        except Exception as e:
            print(f"\nErro ao chamar a LLM no cluster {cluster}: {e}")
            pass
            
    print("\nRefinamento LLM concluído.")
    
    new_templates = []
    for i, row in dataset.iterrows():
        c_id = row['Cluster']
        if c_id in refined_templates_map:
            new_templates.append(refined_templates_map[c_id])
        else:
            new_templates.append(row['Template'])
            
    dataset['Template'] = new_templates
