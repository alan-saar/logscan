import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np

# dummy data
data = ["Log content number " + str(i) for i in range(1000)]
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(data)

print(type(vectors))
clusterModel = DBSCAN(min_samples=2)
clusterModel.fit(vectors)
print(clusterModel.labels_[:10])
