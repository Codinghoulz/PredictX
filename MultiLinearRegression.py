import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def predict(x,w,b):
  f_wb=np.dot(x,w)+b   #x_train must be a vector if it is a matrix then it will give a vector of predictions
  return f_wb                 #if x_train is a matrix then f_wb wd be a vector

def compute_cost(x,y,w,b,lamda=1):
  cost=0
  m=x.shape[0]
  err=predict(x,w,b) - y
  err=err**2
  cost=(np.sum(err))/(2*m)
  
  reg_cost = 0
  reg_cost = w**2                                        
  reg_cost = (lamda/(2*m))*np.sum(reg_cost)
  
  return cost+reg_cost

def compute_gradient(x,y,w,b,lamda=1):
  m=len(y)
  predictions = predict(x,w,b)
  err= predictions -y 
  dj_db= np.sum(err)/m
  dj_dw= np.dot(x.T,err)/m   #as there are multiple values of w so it is calculating different gradients for different parameters
  
  dj_dw_reg=(lamda/m)*w
  
  return (dj_dw+dj_dw_reg),dj_db

def update_weights(w,b,alpha,x,y):
  m,n=x.shape
  dj_dw,dj_db=compute_gradient(x,y,w,b)
  
  w=w-alpha*dj_dw
  b=b-alpha*dj_db
  return w,b

def train(x,y,w,b,alpha,num_iters):
  j_hist=[]
  for i in range(num_iters):
    w,b=update_weights(w,b,alpha,x,y)
    if(i%100==0):
      j=compute_cost(x,y,w,b)
      print(f'iteration:{i:4}  cost:{j}')
    if i<100000:
      j_hist.append(compute_cost(x,y,w,b))

  return w,b

