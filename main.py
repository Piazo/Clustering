import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff


databrut = arff.loadarff(open("xclara.arff", 'r'))
data = [[x[0], x[1]] for x in databrut[0]]

# Affichage en 2D
f0 = [f[0] for f in data]
f1 = [f[1] for f in data]

plt.scatter(f0, f1, s=8)
plt.title("Données initiales")
plt.show