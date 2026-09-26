import numpy as np

#Creating linear regression model from scratch

class LinearRegression():
    def __init__(self, learning_rate: float = 0.01, no_of_iterations: int = 1000):
        if learning_rate <= 0: 
            raise ValueError("learning_rate must be greater than 0.") 
        if no_of_iterations <= 0: 
            raise ValueError("no_of_iterations must be greater than 0.")
        
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
    
    def fit(self, X, Y):
        self.no_of_training_examples, self.no_of_features = X.shape
        
        #initiating weight and bias
        self.weight = np.zeros(self.no_of_features)
        self.bias = 0
        self.X = X
        self.Y = Y
        
        #implementing gradient descent algorithm
        for i in range(self.no_of_iterations):
            self.update_params()
    
    def update_params(self):
        Y_prediction = self.predict(self.X)
        
        # calculate gradients

        d_weight = - (2 * (self.X.T).dot(self.Y - Y_prediction)) / self.no_of_training_examples
    
        d_bias = - 2 * np.sum(self.Y - Y_prediction)/self.no_of_training_examples
    
        # upadating the weights
        
        self.weight = self.weight - self.learning_rate*d_weight
        self.bias = self.bias - self.learning_rate*d_bias
    
    
    def predict(self,X):
        
        return X.dot(self.weight)+self.bias