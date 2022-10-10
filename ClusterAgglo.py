from cProfile import label
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from scipy.io import arff
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

databrut = arff.loadarff(open("./data/xclara.arff", 'r'))
data = [[x[0], x[1]] for x in databrut[0]]
f0 = [f[0] for f in data]
f1 = [f[1] for f in data]

# Donnees dans datanp
# print("Dendrogramme 'single' donnees initiales " )
# linked_mat = shc.linkage(data, 'single')
# plt.figure(figsize = (12 ,12))
# shc . dendrogram ( linked_mat ,
#     orientation = 'top',
#     distance_sort = 'descending',
#     show_leaf_counts = False )
# plt . show ()

tps1 = time.time()
model = cluster.AgglomerativeClustering(distance_threshold=10, linkage='single', n_clusters=None)
model = model.fit(data)
tps2 = time.time()

labels=model.labels_
k = model.n_clusters_
leaves = model.n_leaves_

plt.scatter(f0,f1,c=labels, s=8)
plt.title("Resultat du clustering")
plt.show()
print ( " nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )

# set the number of clusters
k = 4
tps1 = time . time ()
model = cluster .AgglomerativeClustering( linkage = 'single' , n_clusters = k )
model = model . fit (data)
tps2 = time . time ()
labels = model . labels_
kres = model . n_clusters_
leaves = model . n_leaves_