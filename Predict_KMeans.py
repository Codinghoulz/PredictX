import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from kMeans import *

data=pd.read_csv('unsupervised_data.csv')
X=data.values

ratio=0.8
x_train,y_train,x_test,y_test=split(X,ratio)

init_centroids=kMeans_init_centroids(x_train, 4)
centroids_fin,idx_fin=run_kMeans(x_train,init_centroids,max_iters=10, plot_progress=True)
y_pred=find_closest_centroids(x_test,centroids_fin)
print(y_pred)