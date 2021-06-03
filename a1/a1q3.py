# run ./q3.sh
# target files have prefix a1q3 and file type png
import numpy
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

X_test_A = numpy.genfromtxt('X_test_A.csv', delimiter=',')
X_test_B = numpy.genfromtxt('X_test_B.csv', delimiter=',')
X_test_C = numpy.genfromtxt('X_test_C.csv', delimiter=',')

X_train_A = numpy.genfromtxt('X_train_A.csv', delimiter=',')
X_train_B = numpy.genfromtxt('X_train_B.csv', delimiter=',')
X_train_C = numpy.genfromtxt('X_train_C.csv', delimiter=',')

Y_test_A = numpy.genfromtxt('Y_test_A.csv', delimiter=',')
Y_test_B = numpy.genfromtxt('Y_test_B.csv', delimiter=',')
Y_test_C = numpy.genfromtxt('Y_test_C.csv', delimiter=',')

Y_train_A = numpy.genfromtxt('Y_train_A.csv', delimiter=',')
Y_train_B = numpy.genfromtxt('Y_train_B.csv', delimiter=',')
Y_train_C = numpy.genfromtxt('Y_train_C.csv', delimiter=',')

def linReg(X, y, testX, testy):
    reg = LinearRegression().fit(X, y)
    pre = reg.predict(testX)
    return reg.coef_, mean_squared_error(pre, testy)

def ridge(X, y, testX, testy, reg):
    clf = Ridge(alpha = reg)
    clf.fit(X, y)
    pre = clf.predict(testX)
    return clf.coef_, mean_squared_error(testy, pre)

def lasso(X, y, testX, testy, reg):
    # since lasso optimization objective is (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    # 
    n = len(y)
    clf = Lasso(alpha = reg/n)
    clf.fit(X, y)
    pre = clf.predict(testX)
    return clf.coef_, mean_squared_error(testy, pre)

def setA():
    wLinReg, resLinReg = linReg(X_train_A, Y_train_A, X_test_A, Y_test_A)
    print("Average Mean Squared Error for Set A using Linear Regression is:", resLinReg)
    wRidge1, resRidge1 = ridge(X_train_A, Y_train_A, X_test_A, Y_test_A, 1)
    print("Average Mean Squared Error for Set A using Ridge Regression 1 is:", resRidge1)
    wRidge10, resRidge10 = ridge(X_train_A, Y_train_A, X_test_A, Y_test_A, 10)
    print("Average Mean Squared Error for Set A using Ridge Regression 10 is:", resRidge10)
    wLasso1, resLasso1 = lasso(X_train_A, Y_train_A, X_test_A, Y_test_A, 1)
    print("Average Mean Squared Error for Set A using Lasso 1 is:", resLasso1)
    wLasso10, resLasso10 = lasso(X_train_A, Y_train_A, X_test_A, Y_test_A, 10)
    print("Average Mean Squared Error for Set A using Lasso 10 is:", resLasso10)
    best = min(resLinReg, resRidge1, resRidge10, resLasso1, resLasso10)
    print("Best performer of set A is: ", end='')
    if (best == resLinReg):
        print("Linear Regression")
    elif (best == resRidge1):
        print("Ridge Regression 1")
    elif (best == resRidge10):
        print("Ridge Regression 10")
    elif (best == resLasso1):
        print("Lasso 1")
    elif (best == resLasso10):
        print("Lasso 10")
    kwargs = dict(histtype='step', alpha = 0.2, bins=40)
    figA = plt.figure()
    plt.title("A1 E3 Set A")
    plt.hist(wLinReg, **kwargs, label = 'Linear Regression')
    plt.hist(wRidge1, **kwargs, label = 'Ridge Regression 1')
    plt.hist(wRidge10, **kwargs, label = 'Ridge Regression 10')
    plt.hist(wLasso1, **kwargs, label = 'Lasso 1')
    plt.hist(wLasso10, **kwargs, label = 'Lasso 10')
    plt.legend(loc='upper right')
    plt.title
    figA.savefig('a1q3A.png', facecolor='w', edgecolor='w')
    

def setB():
    wLinReg, resLinReg = linReg(X_train_B, Y_train_B, X_test_B, Y_test_B)
    print("Average Mean Squared Error for Set B using Linear Regression is:", resLinReg)
    wRidge1, resRidge1 = ridge(X_train_B, Y_train_B, X_test_B, Y_test_B, 1)
    print("Average Mean Squared Error for Set B using Ridge Regression 1 is:", resRidge1)
    wRidge10, resRidge10 = ridge(X_train_B, Y_train_B, X_test_B, Y_test_B, 10)
    print("Average Mean Squared Error for Set B using Ridge Regression 10 is:", resRidge10)
    wLasso1, resLasso1 = lasso(X_train_B, Y_train_B, X_test_B, Y_test_B, 1)
    print("Average Mean Squared Error for Set B using Lasso 1 is:", resLasso1)
    wLasso10, resLasso10 = lasso(X_train_B, Y_train_B, X_test_B, Y_test_B, 10)
    print("Average Mean Squared Error for Set B using Lasso 10 is:", resLasso10)
    best = min(resLinReg, resRidge1, resRidge10, resLasso1, resLasso10)
    print("Best performer of set B is: ", end='')
    if (best == resLinReg):
        print("Linear Regression")
    elif (best == resRidge1):
        print("Ridge Regression 1")
    elif (best == resRidge10):
        print("Ridge Regression 10")
    elif (best == resLasso1):
        print("Lasso 1")
    elif (best == resLasso10):
        print("Lasso 10")
    figB = plt.figure()
    plt.title("A1 E3 Set B")
    kwargs = dict(histtype='step', alpha = 0.2, bins=40)
    plt.hist(wLinReg, **kwargs, label = 'Linear Regression')
    plt.hist(wRidge1, **kwargs, label = 'Ridge Regression 1')
    plt.hist(wRidge10, **kwargs, label = 'Ridge Regression 10')
    plt.hist(wLasso1, **kwargs, label = 'Lasso 1')
    plt.hist(wLasso10, **kwargs, label = 'Lasso 10')
    plt.legend(loc='upper right')
    plt.title
    figB.savefig('a1q3B.png', facecolor='w', edgecolor='w')
    

def setC():
    wLinReg, resLinReg = linReg(X_train_C, Y_train_C, X_test_C, Y_test_C)
    print("Average Mean Squared Error for Set C using Linear Regression is:", resLinReg)
    wRidge1, resRidge1 = ridge(X_train_C, Y_train_C, X_test_C, Y_test_C, 1)
    print("Average Mean Squared Error for Set C using Ridge Regression 1 is:", resRidge1)
    wRidge10, resRidge10 = ridge(X_train_C, Y_train_C, X_test_C, Y_test_C, 10)
    print("Average Mean Squared Error for Set C using Ridge Regression 10 is:", resRidge10)
    wLasso1, resLasso1 = lasso(X_train_C, Y_train_C, X_test_C, Y_test_C, 1)
    print("Average Mean Squared Error for Set C using Lasso 1 is:", resLasso1)
    wLasso10, resLasso10 = lasso(X_train_C, Y_train_C, X_test_C, Y_test_C, 10)
    print("Average Mean Squared Error for Set C using Lasso 10 is:", resLasso10)
    best = min(resLinReg, resRidge1, resRidge10, resLasso1, resLasso10)
    print("Best performer of set C is: ", end='')
    if (best == resLinReg):
        print("Linear Regression")
    elif (best == resRidge1):
        print("Ridge Regression 1")
    elif (best == resRidge10):
        print("Ridge Regression 10")
    elif (best == resLasso1):
        print("Lasso 1")
    elif (best == resLasso10):
        print("Lasso 10")
    figC = plt.figure()
    plt.title("A1 E3 Set C")
    kwargs = dict(histtype='step', alpha = 0.2, bins=40)
    plt.hist(wLinReg, **kwargs, label = 'Linear Regression')
    plt.hist(wRidge1, **kwargs, label = 'Ridge Regression 1')
    plt.hist(wRidge10, **kwargs, label = 'Ridge Regression 10')
    plt.hist(wLasso1, **kwargs, label = 'Lasso 1')
    plt.hist(wLasso10, **kwargs, label = 'Lasso 10')
    plt.legend(loc='upper right')
    plt.title
    figC.savefig('a1q3C.png', facecolor='w', edgecolor='w')
    
setA()
setB()
setC()