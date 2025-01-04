import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from MultiLinearRegression import *

# Load the CSV file into a pandas DataFrame
data = pd.read_csv('linear_regression_train.csv') 
x_train = data.iloc[0:42000, :-1].values  
y_train= data.iloc[0:42000, -1].values  
x_test = data.iloc[42001:48000, :-1].values  
y_test= data.iloc[42001:48000, -1].values 

 
#data scaling
x_train_scaled = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
x_test_scaled= (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)

#value initializations
b_init=0
m,n=x_train.shape
alpha=0.1
num_iters=10000
w_init=np.zeros(n)


w_fin,b_fin=train(x_train_scaled,y_train,w_init,b_init,alpha,num_iters)
cost=compute_cost(x_test_scaled,y_test,w_fin,b_fin)
print (f"cost :{cost}")
print(w_fin,b_fin)

#evaluating our results
y_pred=predict(x_test_scaled,w_fin,b_fin)
r2 = r2_score(y_test,y_pred )
print(f"R² score: {r2}")