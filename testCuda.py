import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff
from sklearn import cluster, neighbors
import scipy.cluster.hierarchy as shc

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
import hdbscan


path = "/home/alaverdo/Bureau/5A/MLNS/Clustering/dataset-rapport/"
allfiles = ["x1.txt", "x2.txt", "x3.txt", "x4.txt", 
            "y1.txt", "zz1.txt", "zz2.txt"]
filename = allfiles[5]
df = pd.read_csv(path+filename, sep = " ", skipinitialspace=True)

# Data preprocessing
scaler = MinMaxScaler().fit(df)
data = scaler.transform(df).tolist()

# 1ère dimension, axe des abscisses
f0 = [f[0] for f in data]
# 2ème dimension, axe des ordonnées
f1 = [f[1] for f in data]

def kmed():
    print("Appel KMed pour une valeur de k allant de 2 a 25")
    distmatrix = euclidean_distances(data)
    minLoss = 0
    kOpti = 0
    optiLabels = []
    optiIter = 0
    for k in range(2, 25):
        print("appel pour k =",k)
        fp = kmedoids.fasterpam(distmatrix, k)
        iter_kmed = fp.n_iter
        labels_kmed = fp.labels
        score = silhouette_score(data, labels_kmed)
        if(score > minLoss):
            minLoss = score
            kOpti = k
            optiLabels = labels_kmed
            optiIter = iter_kmed
    
    print("Loss with FasterPAM : " , minLoss)
    print("nb clusters = ", kOpti, " --- nb iter = ", optiIter)
    
    plt.scatter(f0, f1, c=optiLabels, s=8)
    plt.title("Données après clustering kmedoids")
    plt.show()

def kmean():
    print("Appel KMeans pour une valeur de k allant de 2 a 25")
    
    minLoss = 0
    kOpti = 0
    optiLabels = []
    # nombre de cluster prédéfini
    for k in range(2, 25):
        print("appel pour k =",k)
        model = cluster.KMeans(n_clusters=k, init='k-means++')
        model.fit(data)
        labels = model.labels_
        iterations = model.n_iter_
        score = silhouette_score(data, labels)
        if(score > minLoss):
            minLoss = score
            kOpti = k
            optiLabels = labels
    
    tps1 = time.time()
    #Création du model
    model = cluster.KMeans(n_clusters=kOpti, init='k-means++')
    model.fit(data)
    tps2 = time.time()
    
    labels = model.labels_
    iterations = model.n_iter_


    print("nb clusters = ", kOpti, "nb iter = ", iterations)
    print("runtime = ", round((tps2-tps1)*1000,2) ,  "ms ")

    print("Silhouette Coefficient: ", silhouette_score(data, labels))
    print("Davies Bouldin score: ", davies_bouldin_score(data, labels))
    
    plt.scatter(f0, f1, c=labels, s=4)
    plt.title("Données après clustering kmedoids")
    plt.show()
    
    
def clusterAglo():
    print("Appel clusterAglo pour une valeur de k allant de 2 a 25")
    minLoss = 0
    kOpti = 0
    optiLabels = []
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


def dbscan():
    print("Appel dbscan")
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
    best_eps = 0
    # nous devons commencer à 2 car le silhouette_score prend minimum 2 clusters
    
    
    
    # On selectionne les quartiles pour les valeurs de eps qu'on sohaite selectionner
    quartileSelection = np.linspace(0.8, 1, num=20)
    tabEps = np.quantile(trie, quartileSelection)
    print(tabEps)
    for j in range(20, 30):
        print("j = ", j)
        for i in tabEps:
            model = cluster.DBSCAN(eps=i, min_samples=j)
            model.fit(data) 
            labels = model.labels_
            try:
                score = silhouette_score(data, labels)
                if((max(labels)+1 > 1 and (max(labels)+1<1000)) and (score > max_silhouette)):
                    bestMinSamples = j
                    max_silhouette = score
                    labels_used = labels
                    best_eps = i
            except:
                pass
    
    tps1 = time.time()
    model = cluster.DBSCAN(eps=i, min_samples=j)
    model.fit(data) 
    tps2 = time.time()


    print("nb eps trouvé = ", best_eps)
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
    


def hdbscan():
    pass










if __name__ == "__main__":
    dbscan()