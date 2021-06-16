import sys
import numpy as np

#Exercise 3
#Usage: python3 ex3.py X_train Y_train X_test Y_test C eps

def SVR(X_train, Y_train, C, eps):
    #Implement me! You may choose other parameters eta, max_pass, etc. internally
    #Return: parameter vector w, b
    max_pass = 1000
    eta = 0.0001
    n = len(y)
    w = np.zeros(X_train[0])
    b = 0

    for t in range(max_pass):
        for i in range(n):

            if (abs(Y_train[i] - (np.inner(X_train[i],w) + b)) - eps) >= 0:
                if ((Y_train[i] - (np.inner(X_train[i],w) + b)) - eps) >= 0:
                    w = w - np.multiply(eta, X_train[i])
                    b = b - eta
                else:
                    w = w + np.multiply(eta, X_train[i])
                    b = b + eta
        w = np.multiply(w, 1/(1+eta))
    return w, b

def compute_loss(X, Y, w, b, C, eps):
    #Implement me!
    #Return: loss computed on the given set
    pass

def compute_error(X, Y, w, b, C, eps):
    #Implement me!
    #Return: error computed on the given set
    pass


if __name__ == "__main__":
    args = sys.argv[1:]
    #You may import the data some other way if you prefer
    X_train = np.loadtxt(args[0], delimiter=",")
    Y_train = np.loadtxt(args[1], delimiter=",")
    X_test = np.loadtxt(args[2], delimiter=",")
    Y_test = np.loadtxt(args[3], delimiter=",")
    C = float(args[4])
    eps = float(args[5])
