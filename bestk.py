import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff
from sklearn import cluster
from sklearn.metrics import silhouette_score



databrut = arff.loadarff(open("./data/xclara.arff", 'r'))
data = [[x[0], x[1]] for x in databrut[0]]

# Affichage en 2D

# 1ère dimension, axe des abscisses
f0 = [f[0] for f in data]
# 2ème dimension, axe des ordonnées
f1 = [f[1] for f in data]

tps1 = time.time()

max_silhouette = 0
k_used = 0
labels_used = []
# nous devons commencer à 2 car le silhouette_score prend minimum 2 clusters
for k in range(2,30):
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(data) 
    labels = model.labels_
    silhouetteScore = silhouette_score(data, labels)
    if(silhouetteScore > max_silhouette):
        k_used = k
        max_silhouette = silhouetteScore
        labels_used = labels

tps2 = time.time()


print("nb clusters trouvé = ", k_used)
print("runtime = ", round((tps2-tps1)*1000,2) ,  "ms ")

print("Silhouette Coefficient max: ", max_silhouette)

# On plot le cluster

plt.scatter(f0, f1, c=labels_used, s=8)
plt.title("Données après clustering")
plt.show()
