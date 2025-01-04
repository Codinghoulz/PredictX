import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from knn import *

data=pd.read_csv('binary_classification_train.csv')

x_train = data.iloc[0:42000, :-1].values  
y_train= data.iloc[0:42000, -1].values  
x_test = data.iloc[42001:48000, :-1].values  
y_test= data.iloc[42001:48000, -1].values

x_train_scaled = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
x_test_scaled= (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)
k=2


a=predict (x_train,x_test[0],k,y_train)

print (a)