import pandas as pd
from sklearn import cluster
from sklearn import metrics
import kmedoids
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.io import arff

#Dataset valable pour kmeans : 2d-4c.arff avec k=4
#Dataset valable pour kmeans : blobs.arff avec k=3
databrut = arff.loadarff(open("xclara.arff", 'r'))
data = [[x[0], x[1]] for x in databrut[0]]

# Affichage en 2D
f0 = [f[0] for f in data]
f1 = [f[1] for f in data]


############################## Clustering K Means ##############################

"""
print("Appel kmeans")
# On definit un tableau sil qui va stocker les scores de silhouette
sil = []
# On fait varier k de 2 a 60
for k in range(2,60):
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(data)
    labels = model.labels_
    sil.append(silhouette_score(data, labels))

# On affiche le score max trouve et le k associe
print("Score max = " + str(max(sil)) + " pour k = " + str(sil.index(max(sil)) + 2))

# Affichage du plot avec le meilleur score silhouette
model = cluster.KMeans(n_clusters= sil.index(max(sil)) + 2, init='k-means++')
model.fit(data)
labels = model.labels_

plt.scatter(f0,f1, c=labels,s=8)
plt.title("donnees apres clustering kmeans")
plt.show()


time1 = time.time()
print("Temps de calcul silhouette score : " + str(time.time()- time1) + 
      " secondes, score = " + str(silhouette_score(data, labels)))
time1 = time.time()
print("Temps de calcul davies bouldin score : " + str(time.time()- time1) + 
      " secondes, score = " + str(davies_bouldin_score(data, labels)) )
"""

############################## Clustering K Medoid ##############################

sil = []
loss = []
for k in range(2,30):
    distmatrix=euclidean_distances(data)
    fp = kmedoids.fasterpam(distmatrix, k)
    loss.append(fp.loss)
    iter_kmed = fp.n_iter
    labels_kmed = fp.labels
    sil.append(silhouette_score(data, labels_kmed))

# On affiche le score max trouve et le k associe
print("Score max = " + str(max(sil)) + " pour k = " + str(sil.index(max(sil)) + 2))

# Affichage du plot avec le meilleur score silhouette
fp = kmedoids.fasterpam(distmatrix, sil.index(max(sil)) +2 )
iter_kmed = fp.n_iter
labels_kmed = fp.labels

plt.plot(loss)
plt.show()
plt.clf()
print("Loss with fasterPAM:", fp.loss)
plt.scatter(f0,f1, c=labels_kmed,s=8)
plt.title("donnees apres clustering kMedoids")
plt.show()





