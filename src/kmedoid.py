import time
from xmlrpc.client import MAXINT
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff
from sklearn import cluster
import kmedoids
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

databrut = arff.loadarff(open("/home/alaverdo/Bureau/5A/MLNS/Clustering/data/dartboard1.arff", 'r'))
data = [[x[0], x[1]] for x in databrut[0]]

# Affichage en 2D

# 1ère dimension, axe des abscisses
f0 = [f[0] for f in data]
# 2ème dimension, axe des ordonnées
f1 = [f[1] for f in data]

distmatrix = euclidean_distances(data)

minLoss = 0
kOpti = 0
optiLabels = []
optiIter = 0
for k in range(2, 20):
    fp = kmedoids.fasterpam(distmatrix, k)
    iter_kmed = fp.n_iter
    labels_kmed = fp.labels
    if(silhouette_score(data, labels_kmed) > minLoss):
        minLoss = silhouette_score(data, labels_kmed)
        kOpti = k
        optiLabels = labels_kmed
        optiIter = iter_kmed


print("Loss with FasterPAM : " , minLoss)

print("nb clusters = ", kOpti, " --- nb iter = ", optiIter)

plt.scatter(f0, f1, c=optiLabels, s=8)
plt.title("Données après clustering kmedoids")
plt.show()