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
            split_vals = set()
            for input in range(num_inputs):
                split_vals.add(self.X[input][feature])

            best_feat_split_loss = float("inf")
            for split in split_vals:
                yl = np.array([])
                yr = np.array([])
                for i in range(num_inputs):
                    if (self.X[i][feature] <= split):
                        yl = np.append(yl, self.y[i])
                    else:
                        yr = np.append(yr, self.y[i])
                split_loss = len(yl)/num_inputs * self.f(yl) + len(yr)/num_inputs * self.f(yr)
                if (split_loss < best_feat_split_loss):
                    best_feat_split_val = split
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
x_axis = range(13)

for i in range(13):
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
w_accuracies = []
wo_accuracies = []
for j in range(11):
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
    pred_y_without = np.empty_like([y_test])
    pred_y_with = np.empty_like([y_test])
    for b in range(101):
        dtEntropyWithout = DecisionTree(entropy, 3)
        dtEntropyWithout.build(datasets_X[b], datasets_y[b])
        entropy_y_test_without = dtEntropyWithout.predict(X_test)
        pred_y_without = np.append(pred_y_without, np.array([entropy_y_test_without]), axis=0)

        dtEntropyWith = DecisionTree(entropy, 3, True)
        dtEntropyWith.build(datasets_X[b], datasets_y[b])
        entropy_y_test_with = dtEntropyWith.predict(X_test)
        pred_y_with = np.append(pred_y_with, np.array([entropy_y_test_with]), axis=0)

    pred_y_without = pred_y_without[1:]
    pred_y_with = pred_y_with[1:]

    without_acc = 0
    with_acc = 0
    test_n = len(y_test)
    for i in range(test_n):
        pred_without = pred_y_without[:,i]
        y_without = round(np.average(pred_without))
        if (y_without == y_test[i]):
            without_acc += 1

        pred_with = pred_y_with[:,i]
        y_with = round(np.average(pred_with))
        if (y_with == y_test[i]):
            with_acc += 1

    without_acc /= test_n
    with_acc /= test_n

    wo_accuracies.append(without_acc)
    w_accuracies.append(with_acc)

print("Without Random Forests")
print("Median: {}, Minimum: {}, Maximum: {}".format(np.median(wo_accuracies), np.amin(wo_accuracies), np.amax(wo_accuracies)))
print("With Random Forests")
print("Median: {}, Minimum: {}, Maximum: {}".format(np.median(w_accuracies), np.amin(w_accuracies), np.amax(w_accuracies)))