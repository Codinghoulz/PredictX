import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polynomial import *

# Load dataset
x = pd.read_csv('polynomial_regression_train.csv')

# Hyperparameters
degree = 2
ratio = 0.8
alpha = 0.01
num_iters = 4000

# Split data into training and test sets
x_train, y_train, x_test, y_test = split(x, ratio)


# Generate polynomial features for training data
x_train_poly = polynomial_features(x_train, degree)
x_test_poly=polynomial_features(x_test,degree)

# Data Scaling (Standardization)
x_train_scaled = ( x_train_poly - np.mean( x_train_poly, axis=0)) / np.std( x_train_poly, axis=0)
x_test_scaled = (x_test_poly - np.mean(x_test_poly, axis=0)) / np.std(x_test_poly, axis=0)

# Initialize weights and bias
w=np.zeros(x_train_scaled.shape[1])
#w = np.random.randn(x_poly.shape[1]) * 0.01 
b = 0


w_final, b_final, j_history = poly_reg(x_train_scaled, y_train.values, w, b, alpha, num_iters)
test_predictions = predict(x_test_scaled, w_final, b_final)

test_cost = compute_cost(x_train_scaled, y_test.values, w_final, b_final)
print(f"Test cost: {test_cost}")

# Optionally: Plot the cost history to observe convergence

plt.plot(j_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost History during Gradient Descent')
plt.show()

#evaluating our results
y_pred=predict(x_test_scaled,w_final,b_final)


r2 = r2_score(y_test,y_pred )
print(f"R² score: {r2}")