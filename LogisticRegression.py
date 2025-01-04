import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score



def sigmoid(z):
    return 1/(1+np.exp(-z))

#predict function
def predict(x,w,b):
    f=np.dot(x,w)+b
    return sigmoid(f)
  
#cost function
def compute_cost(x,y,w,b,lamda=0.001):
    m=x.shape[0]
    predictions = predict(x, w, b)
    loss= (-np.dot(y,np.log(predictions )))- np.dot((1-y),np.log(1-predictions ))
    cost=(np.sum(loss))/m
    
    reg_cost = 0
    reg_cost = w**2                                        
    reg_cost = (lamda/(2*m))*np.sum(reg_cost)
    return cost 
#compute gradient
def compute_gradient(x,y,w,b,lamda=0.001):
    m=x.shape[0]
    predictions=predict(x,w,b)-y
    dj_dw=np.dot(x.T,predictions)/m
    dj_db=np.sum(predict(x,w,b)-y)/m
    
    
    dj_dw_reg=(lamda/m)*w
  
    return (dj_dw),dj_db                              
 
#gradient descent
def gradient_descent(x,y,w,b,alpha,num_iters):
    J_history = []

    for i in range(num_iters):
        dj_dw,dj_db = compute_gradient(x,y,w,b)   
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
        if i<100000:      
            J_history.append( compute_cost(x, y, w, b) )
        if i%100  == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
    return w, b, J_history  




def predict_class(x, w, b):
    probabilities = predict(x, w, b)
    return (probabilities > 0.5).astype(int)  # Classify based on probability threshold of 0.5


def calculate_accuracy(y_true, y_pred):
    """
    Calculate the accuracy of a model by comparing the true labels (y_true) 
    to the predicted labels (y_pred).
    
    Parameters:
    y_true (numpy array): The true class labels (0 or 1)
    y_pred (numpy array): The predicted class labels (0 or 1)
    
    Returns:
    float: The accuracy score (between 0 and 1)
    """
    # Calculate the number of correct predictions
    correct_predictions = np.sum(y_true == y_pred)
    
    # Calculate the total number of predictions
    total_predictions = len(y_true)
    
    # Accuracy is the number of correct predictions divided by the total number of predictions
    accuracy = correct_predictions / total_predictions
    
    return accuracy

