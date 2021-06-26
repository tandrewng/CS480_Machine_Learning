import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
#Load data
X_train = np.loadtxt('data/X_train.csv', delimiter=",")
y_train = np.loadtxt('data/y_train.csv', delimiter=",").astype(int)
X_test = np.loadtxt('data/X_test.csv', delimiter=",")
y_test = np.loadtxt('data/y_test.csv', delimiter=",").astype(int)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
plt.figure()
print(tree.plot_tree(clf,filled=True))