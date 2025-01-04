import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from LogisticRegression import *
#load data
data=pd.read_csv('binary_classification_train.csv')
x_train = data.iloc[0:42000, :-1].values  
y_train= data.iloc[0:42000, -1].values  
x_test = data.iloc[42001:48000, :-1].values  
y_test= data.iloc[42001:48000, -1].values 

x_train_scaled = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
x_test_scaled= (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)

w_in = np.random.randn(x_train.shape[1]) * 0.01  # Small random values
b_in = 0
alpha=0.001
num_iters=10000



a=compute_cost(x_train, y_train, w_in, b_in)
print(a)


w_fin,b_fin,j_hist=gradient_descent(x_train_scaled, y_train, w_in, b_in, alpha, num_iters)
print(w_fin,b_fin)
#for predicting

y_predict=predict_class(x_test_scaled,w_fin,b_fin)
f1 = f1_score(y_test, y_predict)
print(f'F1 Score: {f1}')


# Assuming y_test is the true labels and y_predict is the predicted labels
accuracy = calculate_accuracy(y_test, y_predict)

print(f'Accuracy: {accuracy * 100:.2f}%')
