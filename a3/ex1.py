import numpy as np
import math
import matplotlib.pyplot as plt
import random

from numpy.core.numeric import empty_like

class Node:
    def __init__(self, X, y, depth, max_depth, f, random_forests):
        self.X = X
        self.y = y
        self.depth = depth
        self.max_depth = max_depth
        self.f = f
        self.left = None
        self.right = None
        self.descision = None
        self.random_forests = random_forests
        if (depth < max_depth and not self.isPure()): 
            # internal node, must be split
            self.split()
        else:
            # leaf node makes a descision
            self.descision = round(np.average(self.y))
        return
    
    def split(self):
        d = len(self.X[0])
        rootd = list(range(d))
        if (self.random_forests): 
            num_features = round(math.sqrt(d))
            rootd = list(random.sample(range(d), num_features))
        num_inputs = len(self.y)
        splits = np.array([])
        losses = np.array([])

        for feature in rootd:
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

        X_left = np.empty_like([self.X[0]])
        y_left = np.array([])
        X_right = np.empty_like([self.X[0]])
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

        self.left = Node(X_left[1:], y_left, self.depth + 1, self.max_depth, self.f, self.random_forests)
        self.right = Node(X_right[1:], y_right, self.depth + 1, self.max_depth, self.f, self.random_forests)
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
    def __init__(self, loss_f, max_depth = None, random_forests = False):
        self.max_depth = max_depth
        self.f = loss_f
        self.random_forests = random_forests
        return

    def build(self, X, y):
        self.root = Node(X, y, 0, self.max_depth, self.f, self.random_forests)
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

'''
# Part A
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
'''
# Part B
for j in range(11):
    print("Run:", j)
    datasets_X = np.empty_like([X_train])
    datasets_y = np.empty_like([y_train])

    for i in range(101):
        s_X = np.empty_like([X_train[0]])
        s_y = np.array([])

        n = len(X_train)
        for j in range(n):
            new = random.randint(0, n-1)
            s_X = np.append(s_X, np.array([X_train[new]]), axis=0)
            s_y = np.append(s_y, np.array([y_train[new]]), axis=0)

        datasets_X = np.append(datasets_X, np.array([s_X[1:]]), axis=0)
        datasets_y = np.append(datasets_y, np.array([s_y]), axis=0)

    datasets_X = datasets_X[1:]
    datasets_y = datasets_y[1:]
    test_accuracies_without = np.array([])
    test_accuracies_with = np.array([]) 
    for b in range(101):
        dtEntropyWithout = DecisionTree(entropy, 3)
        dtEntropyWithout.build(datasets_X[b], datasets_y[b])
        entropy_y_test = dtEntropyWithout.predict(X_test)
        test_accuracies_without = np.append(test_accuracies_without, accuracy(entropy_y_test, y_test))

        dtEntropyWith = DecisionTree(entropy, 3, True)
        dtEntropyWith.build(datasets_X[b], datasets_y[b])
        entropy_y_test = dtEntropyWith.predict(X_test)
        test_accuracies_with = np.append(test_accuracies_with, accuracy(entropy_y_test, y_test))
    
    test_median_without = np.median(test_accuracies_without)
    test_minimum_without = np.amin(test_accuracies_without)
    test_maximum_without = np.amax(test_accuracies_without)
    print("Without Random Forests")
    print("Test Median: {}, Test Minimum: {}, Test Maximum: {}".format(test_median_without, test_minimum_without, test_maximum_without))

    test_median_with = np.median(test_accuracies_with)
    test_minimum_with = np.amin(test_accuracies_with)
    test_maximum_with = np.amax(test_accuracies_with)
    print("With Random Forests")
    print("Test Median: {}, Test Minimum: {}, Test Maximum: {}".format(test_median_with, test_minimum_with, test_maximum_with))
