import numpy as np

#Exercise 2
#Usage: python3 ex2.py
#Load the files in some other way (e.g., pandas) if you prefer
X_train_A = np.loadtxt('data/X_train_A.csv', delimiter=",")
Y_train_A = np.loadtxt('data/Y_train_A.csv',  delimiter=",").astype(int)

X_train_B = np.loadtxt('data/X_train_B.csv', delimiter=",")
Y_train_B = np.loadtxt('data/Y_train_B.csv', delimiter=",").astype(int)
X_test_B = np.loadtxt('data/X_test_B.csv', delimiter=",")
Y_test_B = np.loadtxt('data/Y_test_B.csv', delimiter=",").astype(int)

