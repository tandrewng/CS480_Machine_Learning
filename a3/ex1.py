import numpy as np
import math
import matplotlib.pyplot as plt

from numpy.core.numeric import empty_like

class Node:
    def __init__(self, X, y, depth, max_depth, f):
        self.X = X
        self.y = y
        self.depth = depth
        self.max_depth = max_depth
        self.f = f
        self.left = None
        self.right = None
        self.descision = None
        if (depth < max_depth and not self.isPure()): 
            # internal node, must be split
            self.split()
        else:
            # leaf node makes a descision
            self.descision = round(np.average(self.y))
        return
    
    def split(self):
        num_features = len(self.X[0])
        num_inputs = len(self.y)
        splits = np.array([])
        losses = np.array([])

        for feature in range(num_features):
            # best split val for the feature
            best_feat_split_val = self.X[0][feature]
            best_feat_split_loss = float("inf")
            for input in range(num_inputs):
                split_val = self.X[input][feature]
                yl = np.array([])
                yr = np.array([])
                for i in range(num_inputs):
                    if (self.X[i][feature] <= split_val):
                        yl = np.append(yl, self.y[i])
                    else:
                        yr = np.append(yr, self.y[i])
                split_loss = len(yl)/num_inputs * self.f(yl) + len(yr)/num_inputs * self.f(yr)
                if (split_loss < best_feat_split_loss):
                    best_feat_split_val = split_val
                    best_feat_split_loss = split_loss
            splits = np.append(splits, best_feat_split_val)
            losses = np.append(losses, best_feat_split_loss)

        best_split_feature = np.argmin(losses)
        # print(best_split_feature)
        # print(splits)
        # print(losses)
        # print(len(self.X))

        X_left = np.empty_like([splits])
        y_left = np.array([])
        X_right = np.empty_like([splits])
        y_right = np.array([])

        for i in range(num_inputs):
            if (self.X[i][best_split_feature] <= splits[best_split_feature]):
                X_left = np.append(X_left, np.array([self.X[i]]), axis=0)
                y_left = np.append(y_left, self.y[i])
            else :
                X_right = np.append(X_right, np.array([self.X[i]]), axis=0)
                y_right = np.append(y_right, self.y[i])

        # if left or right split is empty, then we want the node to guess the opposite of the mode of y
        if (len(y_left) == 0):
            y_left = np.append(y_left,(1 - round(np.average(self.y))))

        if (len(y_right) == 0):
            y_right = np.append(y_right,(1 - round(np.average(self.y))))

        self.left = Node(X_left[1:], y_left, self.depth + 1, self.max_depth, self.f)
        self.right = Node(X_right[1:], y_right, self.depth + 1, self.max_depth, self.f)
        self.split_feature = best_split_feature
        self.split_val = splits[best_split_feature]
        return

    def isPure(self):
        has0 = False
        has1 = False
        for i in self.y:
            if (i == 0):
                has0 = True
            if (i == 1):
                has1 = True
            if (has0 and has1):
                return False
        return True
    
    def predict(self, x):
        if (self.descision is not None):
            return self.descision
        elif (x[self.split_feature] <= self.split_val):
            return self.left.predict(x)
        else:
            return self.right.predict(x)


class DecisionTree:
    #You will likely need to add more arguments to the constructor
    def __init__(self, loss_f, max_depth=None):
        self.max_depth = max_depth
        self.f = loss_f
        return

    def build(self, X, y):
        self.root = Node(X, y, 0, self.max_depth, self.f)
        return
    
    def predict(self, X):
        predictions = np.array([])
        for x in X:
            predictions = np.append(predictions, self.root.predict(x))
        return predictions

def missclassification(y):
    if (len(y) == 0): return 0
    p_hat = 0.
    
    for i in y:
        p_hat += i

    p_hat /= len(y)
    return min(p_hat, (1.-p_hat))

def gini(y):
    if (len(y) == 0): return 0
    p_hat = 0.

    for i in y:
        p_hat += i

    p_hat /= len(y)
    return p_hat*(1.-p_hat)

def entropy(y):
    if (len(y) == 0): return 0
    p_hat = 0.

    for i in y:
        p_hat += i

    p_hat /= len(y)

    if(p_hat == 0):
        return - (1.-p_hat) * math.log((1.-p_hat), 2)
    if(p_hat == 1):
        return (0.-p_hat) * math.log(p_hat, 2)
    return (0.-p_hat) * math.log(p_hat, 2) - (1.-p_hat) * math.log((1.-p_hat), 2)

def accuracy (y_guess, y_actual):
    retval = 0.
    inputs = len(y_guess)
    for i in range(inputs):
        if (y_guess[i] == y_actual[i]):
            retval += 1.
    retval /= inputs
    return retval

#Load data
X_train = np.loadtxt('data/X_train.csv', delimiter=",")
y_train = np.loadtxt('data/y_train.csv', delimiter=",").astype(int)
X_test = np.loadtxt('data/X_test.csv', delimiter=",")
y_test = np.loadtxt('data/y_test.csv', delimiter=",").astype(int)

missclassification_train = np.array([])
missclassification_test = np.array([])
gini_train = np.array([])
gini_test = np.array([])
entropy_train = np.array([])
entropy_test = np.array([])
x_axis = range(14)

for i in range(14):
    dtMissclassification = DecisionTree(missclassification, i)
    dtMissclassification.build(X_train, y_train)
    missclassification_y_train = dtMissclassification.predict(X_train)
    missclassification_train = np.append(missclassification_train, accuracy(missclassification_y_train, y_train))
    missclassification_y_test = dtMissclassification.predict(X_test)
    missclassification_test = np.append(missclassification_test, accuracy(missclassification_y_test, y_test))

    dtGini = DecisionTree(gini, i)
    dtGini.build(X_train, y_train)
    gini_y_train = dtGini.predict(X_train)
    gini_train = np.append(gini_train, accuracy(gini_y_train, y_train))
    gini_y_test = dtGini.predict(X_test)
    gini_test = np.append(gini_test, accuracy(gini_y_test, y_test))

    dtEntropy = DecisionTree(entropy, i)
    dtEntropy.build(X_train, y_train)
    entropy_y_train = dtEntropy.predict(X_train)
    entropy_train = np.append(entropy_train, accuracy(entropy_y_train, y_train))
    entropy_y_test = dtEntropy.predict(X_test)
    entropy_test = np.append(entropy_test, accuracy(entropy_y_test, y_test))

plt.plot(x_axis, missclassification_train, label = "Missclassification Train")
plt.plot(x_axis, missclassification_test, label = "Missclassification Test")
plt.legend()
plt.savefig("Missclassification.png")
plt.clf()

plt.plot(x_axis, gini_train, label = "Gini Train")
plt.plot(x_axis, gini_test, label = "Gini Test")
plt.legend()
plt.savefig("Gini.png")
plt.clf()

plt.plot(x_axis, entropy_train, label = "Entropy Train")
plt.plot(x_axis, entropy_test, label = "Entropy Test")
plt.legend()
plt.savefig("Entropy.png")
plt.clf()