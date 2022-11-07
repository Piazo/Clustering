import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff
from sklearn import cluster, neighbors
from sklearn.metrics import silhouette_score
from numba import jit, cuda
 
def dbscan():
    databrut = arff.loadarff(open("./data/dartboard2.arff", 'r'))
    data = [[x[0], x[1]] for x in databrut[0]]

    # Affichage en 2D

    # 1ère dimension, axe des abscisses
    f0 = [f[0] for f in data]
    # 2ème dimension, axe des ordonnées
    f1 = [f[1] for f in data]

    tps1 = time.time()


    # Distances k plus proches voisins
    # Donnees dans X
    k=5
    neigh = neighbors.NearestNeighbors(n_neighbors = k)
    neigh.fit(data)
    distances , indices = neigh.kneighbors(data)
    # retirer le point " origine "
    newDistances = np.asarray([np.average(distances[i][1:]) for i in range (0, distances.shape[0])])
    trie = np.sort(newDistances )
    plt.title("Plus proches voisins (5)")
    plt.xlabel("nombre total de points")
    plt.ylabel("distance")
    plt.plot(trie)
    plt.show()

    max_silhouette = 0
    bestMinSamples = 0
    labels_used = []
    # nous devons commencer à 2 car le silhouette_score prend minimum 2 clusters
        
    for j in range(1, 15):
        model = cluster.DBSCAN(eps=0.02, min_samples=j)
        model.fit(data) 
        labels = model.labels_
        if((max(labels)+1 > 1 and (max(labels)+1<1000)) and (silhouette_score(data, labels) > max_silhouette)):
            bestMinSamples = j
            max_silhouette = silhouette_score(data, labels)
            labels_used = labels

    tps2 = time.time()


    print("nb eps trouvé = ", 0.02)
    print("nb min_samples trouvé = ", bestMinSamples)
    print("runtime = ", round((tps2-tps1)*1000,2) ,  "ms ")

    print("Silhouette Coefficient max: ", max_silhouette)


    print("nb cluster trouvé: ", max(labels_used)+1)

    nbBruit = 0
    for i in labels_used:
        if i==-1: nbBruit+=1

    print("nb bruit trouvé: ", nbBruit)


    # On plot le cluster


    plt.scatter(f0, f1, c=labels_used, s=8)
    plt.title("Données après clustering")
    plt.show()

dbscan()