import numpy as np
import math

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
            for input in range(1, num_inputs):
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
    return (0.-p_hat) * math.log(p_hat, 2.) - (1.-p_hat) * math.log((1.-p_hat), 2)

#Load data
X_train = np.loadtxt('data/X_train.csv', delimiter=",")
y_train = np.loadtxt('data/y_train.csv', delimiter=",").astype(int)
X_test = np.loadtxt('data/X_test.csv', delimiter=",")
y_test = np.loadtxt('data/y_test.csv', delimiter=",").astype(int)

dtGini = DecisionTree(gini, 2)
dtGini.build(X_train, y_train)
print(dtGini.predict(X_test))