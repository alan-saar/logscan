import pandas as pd
data = pd.DataFrame({"CleanLog": ["A", "B", "A", "C", "B"]})
unique_clean_logs = data['CleanLog'].drop_duplicates().reset_index(drop=True)
cluster_labels = [0, 1, 2] # corresponding to A, B, C
cluster_map = dict(zip(unique_clean_logs, cluster_labels))
data['Cluster'] = data['CleanLog'].map(cluster_map)
print(data)
