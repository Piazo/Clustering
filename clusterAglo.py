import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff
from sklearn import cluster
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc


databrut = arff.loadarff(open("/home/alaverdo/Bureau/5A/MLNS/Clustering/data/xclara.arff", 'r'))
data = [[x[0], x[1]] for x in databrut[0]]

# Affichage en 2D

# 1ère dimension, axe des abscisses
f0 = [f[0] for f in data]
# 2ème dimension, axe des ordonnées
f1 = [f[1] for f in data]


print("Dendrogramme 'single' données initiales")

linked_mat = shc.linkage(data, 'single')

# plt.figure(figsize=(12,12))
# shc.dendrogram(linked_mat, orientation='top', distance_sort='descending', show_leaf_counts=False)
# plt.show()

# Set distance_threshold (0 ensures we compute the full tree)

minLoss = 0
kOpti = 0
optiLabels = []
optiIter = 0
for k in range(2, 25):
    print("appel pour k =",k)
    model = cluster.AgglomerativeClustering(distance_threshold=None, linkage='average', n_clusters=k)
    model = model.fit(data)
    labels = model.labels_
    score = silhouette_score(data, labels)
    if(score > minLoss):
        minLoss = score
        kOpti = k
        optiLabels = labels
        optiIter = iter_kmed
        
        
tps1 = time.time()
model = cluster.AgglomerativeClustering(distance_threshold=None, linkage='average', n_clusters=kOpti)
model = model.fit(data)
tps2 = time.time()

labels = model.labels_
print("score = ",silhouette_score(data, labels))
k = model.n_clusters_
leaves = model.n_leaves_

# Affichage clustering
plt.scatter(f0, f1, c=labels, s=4)
plt.title("Résultat du clustering")
plt.show()

print("nb clusters = ", k, "nb feuilles = ", leaves)
print("runtime = ", round((tps2-tps1)*1000, 2), " ms")




