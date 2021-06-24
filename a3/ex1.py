import numpy as np

class DecisionTree:
    #You will likely need to add more arguments to the constructor
    def __init__(self):
        #Implement me!
        return

    def build(self, X, y):
        #Implement me!
        return
    
    def predict(self, X):
        #Implement me!
        return 

#Load data
X_train = np.loadtxt('data/X_train.csv', delimiter=",")
y_train = np.loadtxt('data/y_train.csv', delimiter=",").astype(int)
X_test = np.loadtxt('data/X_test.csv', delimiter=",")
y_test = np.loadtxt('data/y_test.csv', delimiter=",").astype(int)

