import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score



def calculate_dist(x_train,x_test):
    dist=np.sum((x_train-x_test)**2,axis=1,keepdims=True)
    return dist

def k_closest_points(x_train,x_test,k,y_train):
   # Calculate the distance from each test point to all training points
    distances = calculate_dist(x_train, x_test)
    
    # Get the indices of the k smallest distances
    k_indices = np.argsort(distances)[:k]
    
    # Get the labels for the k closest points
    k_labels = y_train[k_indices]
    
    return k_labels

def predict(x_train,x_test,k,y_train):
    score=k_closest_points(x_train,x_test,k,y_train)
    c1=0
    c2=0
    for i in range (k):
        if(score[i]==0):
            c1=c1+1
            break
        c2=c2+1
     
    if(c2>c1):
        return 1
    return 0
