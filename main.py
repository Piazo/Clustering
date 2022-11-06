import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff
from sklearn import cluster

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score


databrut = arff.loadarff(open("./data/xclara.arff", 'r'))
data = [[x[0], x[1]] for x in databrut[0]]

# Affichage en 2D

# 1ère dimension, axe des abscisses
f0 = [f[0] for f in data]
# 2ème dimension, axe des ordonnées
f1 = [f[1] for f in data]


print("Appel KMeans pour une valeur fixée de k")

tps1 = time.time()

# nombre de cluster prédéfini
k = 3

#Création du model
model = cluster.KMeans(n_clusters=k, init='k-means++')
model.fit(data)

tps2 = time.time()

labels = model.labels_
iterations = model.n_iter_

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Données après clustering")
plt.show()

print("nb clusters = ", k, "nb iter = ", iterations)
print("runtime = ", round((tps2-tps1)*1000,2) ,  "ms ")

print("Silhouette Coefficient: ", silhouette_score(data, labels))
print("Davies Bouldin score: ", davies_bouldin_score(data, labels))