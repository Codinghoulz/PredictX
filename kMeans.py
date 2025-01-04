import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def split(x,ratio):
    m=x.shape[0]
    a=int(m*ratio)
    x_train= x[:a,:-1]
    y_train= x[:a,-1]
    x_test= x[a:m,:-1]
    y_test= x[a:m,-1]
    return x_train,y_train,x_test,y_test


def kMeans_init_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]] 
    #centroids = X.iloc[randidx[:K]].values(for pandas sf)
    
    return centroids

def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    for j in range(K):
        min_distance = float('inf')  # Initialize to a very large number
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - centroids[j]) ** 2)
            if dist < min_distance:
                min_distance = dist
                idx[i] = j  
                    
    return idx

def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    
    for i in range(K):
        sum=0
        c=0
        for j in range(m):
            if(idx[j]==i):
                c=c+1
                sum+=X[j]
        centroids[i]=sum/c
        
    return centroids

def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    

    for i in range(max_iters):
        
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx

