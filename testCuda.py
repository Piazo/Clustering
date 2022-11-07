import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff
from sklearn import cluster, neighbors
import scipy.cluster.hierarchy as shc

import kmedoids
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
import hdbscan
from numpy import zeros,array


path = "/home/alaverdo/Bureau/5A/MLNS/Clustering/dataset-rapport/"
allfiles = ["x1.txt", "x2.txt", "x3.txt", "x4.txt", 
            "zz1.txt", "zz2.txt"]


def kmed():
    print("Appel KMed pour une valeur de k allant de 2 a 25")
    distmatrix = euclidean_distances(data)
    minLoss = 0
    kOpti = 0
    optiLabels = []
    optiIter = 0
    computeTime = 0
    for k in range(2, 25):
        tps1 = time.time()
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
            tps2 = time.time()
            computeTime = tps2 - tps1
    
    print("Loss with FasterPAM : " , minLoss)
    print("nb clusters = ", kOpti, " --- nb iter = ", optiIter)
    print("runtime = ", round(computeTime*1000,2) ,  "ms")
    plt.scatter(f0, f1, c=optiLabels, s=8)
    plt.title("Resultat KMedoids")
    plt.savefig("/home/alaverdo/Bureau/5A/MLNS/Clustering/images/Kmedoids_"+filename+".png")
    plt.show()
    return [round(computeTime*1000,2), minLoss]


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
    
    plt.scatter(f0, f1, c=labels, s=4)
    plt.title("Resultat KMeans")
    plt.savefig("/home/alaverdo/Bureau/5A/MLNS/Clustering/images/Kmeans_"+filename+".png")
    plt.show()
    return [round((tps2-tps1)*1000,2), silhouette_score(data, labels)]
    
    
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
    plt.title("Resultat Cluster agglomeratif")
    plt.savefig("/home/alaverdo/Bureau/5A/MLNS/Clustering/images/ClusterAgglo_"+filename+".png")
    plt.show()

    print("nb clusters = ", k, "nb feuilles = ", leaves)
    print("runtime = ", round((tps2-tps1)*1000, 2), " ms")
    return [round((tps2-tps1)*1000,2), silhouette_score(data, labels)]


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
    plt.title("Resultat DBSCAN")
    plt.savefig("/home/alaverdo/Bureau/5A/MLNS/Clustering/images/DBSCAN_"+filename+".png")
    plt.show()
    return [round((tps2-tps1)*1000,2), max_silhouette]



def hdbscanMod():    
    minLoss = 0
    kOpti = 0
    optiLabels = []
    for k in range(5, 151,5):
        print("appel pour k =",k)
        model = hdbscan.HDBSCAN(min_cluster_size=k)
        model = model.fit(data)
        labels = model.labels_
        score = silhouette_score(data, labels)
        if(score > minLoss):
            minLoss = score
            kOpti = k
            optiLabels = labels
    
    
    tps1 = time.time()
    model = hdbscan.HDBSCAN(min_cluster_size=kOpti)
    model.fit(data) 
    tps2 = time.time()

    labels_used = model.labels_
    max_silhouette = silhouette_score(data, labels_used)
    
    print("runtime = ", round((tps2-tps1)*1000,2) ,  "ms ")

    print("Silhouette Coefficient max :", max_silhouette, "pour min_cluster_size = ", kOpti)
    print("nb cluster trouvé :", max(labels_used)+1)

    nbBruit = 0
    for i in labels_used:
        if i==-1: nbBruit+=1
    print("nb bruit trouvé: ", nbBruit)
    # On plot le cluster
    plt.scatter(f0, f1, c=labels_used, s=4)
    plt.title("Resultat HDBSCAN")
    plt.savefig("/home/alaverdo/Bureau/5A/MLNS/Clustering/images/HDBSCAN_"+filename+".png")
    plt.show()
    return [round((tps2-tps1)*1000,2), max_silhouette]



if __name__ == "__main__":
    tabTempsScore = zeros((5,len(allfiles)))
    i = 0
    for filename in allfiles:
        print(filename)
        df = pd.read_csv(path+filename, sep = " ", skipinitialspace=True)

        # Data preprocessing
        scaler = MinMaxScaler().fit(df)
        data = scaler.transform(df).tolist()

        # 1ère dimension, axe des abscisses
        f0 = [f[0] for f in data]
        # 2ème dimension, axe des ordonnées
        f1 = [f[1] for f in data]
        
        tabTempsScore[0,i] = kmed()
        tabTempsScore[1,i] = kmean()
        tabTempsScore[2,i] = clusterAglo()
        tabTempsScore[3,i] = dbscan()
        tabTempsScore[4,i] = hdbscanMod()
        print(tabTempsScore)
        i+=1
    print(tabTempsScore)