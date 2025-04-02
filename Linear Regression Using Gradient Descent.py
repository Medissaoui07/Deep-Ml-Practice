import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
	
	m, n = X.shape
	theta = np.zeros((n, 1))
	y = y.reshape(-1 , 1)

	for _ in range (iterations) :
		y_predicted = X @ theta 
		error = y_predicted - y 
		gradient = (1/m)*(X.T @ error)
		theta = theta - (alpha*gradient)
	theta_r = np.round(theta , 4)

	return theta_r
