import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function to split data into training and test sets
def split(x, ratio):
    m = x.shape[0]
    a = int(m * ratio)
    x_train = x.iloc[:a, :-1]
    y_train = x.iloc[:a, -1]
    x_test = x.iloc[a:m, :-1]
    y_test = x.iloc[a:m, -1]
    return x_train, y_train, x_test, y_test




# Function to generate polynomial features
def polynomial_features(X, degree):
    X = X.values
    X_poly = X
    n_samples, n_features = X.shape

    for d in range(2, degree + 1):
        for i in range(n_features):
            for j in range(i, n_features):
                X_poly = np.hstack([X_poly, (X[:, i] * X[:, j]).reshape(-1, 1)])
                if i == j:
                    X_poly = np.hstack([X_poly, (X[:, i]**2).reshape(-1, 1)])

    return X_poly


# Prediction function
def predict(x, w, b):
    return np.dot(x, w) + b

# Compute cost function (MSE)
def compute_cost(x,y,w,b,lamda=0.1):
  cost=0
  m=x.shape[0]
  err=predict(x,w,b) - y
  err=err**2
  cost=(np.sum(err))/(2*m)
  
  reg_cost = 0
  reg_cost = w**2                                        #scalar
  reg_cost = (lamda/(2*m))*np.sum(reg_cost)
  
  return cost+reg_cost

# Compute gradients (derivatives of the cost function with respect to w and b)
def compute_gradient(x,y,w,b,lamda=0.1):
  m=len(y)
  predictions = predict(x,w,b)
  err= predictions -y  #here b is added to each term
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

# Update weights and bias using gradient descent
def update_weights(w, b, alpha, x, y):
    dj_dw, dj_db = compute_gradient(x, y, w, b)
    w = w - alpha * dj_dw
    b = b - alpha * dj_db
    return w, b

# Polynomial regression using gradient descent
def poly_reg(x, y, w, b, alpha, num_iters):
    j_hist = []

    for i in range(num_iters):
        w, b = update_weights(w, b, alpha, x, y)
        if i % 100 == 0:  # Record cost at every 100th iteration
            j_hist.append(compute_cost(x, y, w, b))
            print(f"Iteration {i}, Cost: {j_hist[-1]}")

    return w, b, j_hist




def r2_score(y_true, y_pred):
    # Calculate total sum of squares (SST)
    total_sum_of_squares = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Calculate residual sum of squares (SSE)
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    
    # Calculate R² score
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    
    return r2
