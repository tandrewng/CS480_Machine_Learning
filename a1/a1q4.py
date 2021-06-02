import numpy
import matplotlib.pyplot as plt
import sys

X_test_D = numpy.genfromtxt('X_test_D.csv', delimiter=',')
X_test_E = numpy.genfromtxt('X_test_E.csv', delimiter=',')
X_test_F = numpy.genfromtxt('X_test_F.csv', delimiter=',')

X_train_D = numpy.genfromtxt('X_train_D.csv', delimiter=',')
X_train_E = numpy.genfromtxt('X_train_E.csv', delimiter=',')
X_train_F = numpy.genfromtxt('X_train_F.csv', delimiter=',')

Y_test_D = numpy.genfromtxt('Y_test_D.csv', delimiter=',')
Y_test_E = numpy.genfromtxt('Y_test_E.csv', delimiter=',')
Y_test_F = numpy.genfromtxt('Y_test_F.csv', delimiter=',')

Y_train_D = numpy.genfromtxt('Y_train_D.csv', delimiter=',')
Y_train_E = numpy.genfromtxt('Y_train_E.csv', delimiter=',')
Y_train_F = numpy.genfromtxt('Y_train_F.csv', delimiter=',')

# Credit to: geeksforgeeks implementation of quickselect
# https://www.geeksforgeeks.org/quickselect-algorithm/
def partition(arr, l, r):
     
    x = arr[r]
    i = l
    for j in range(l, r):
         
        if arr[j] <= x:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
             
    arr[i], arr[r] = arr[r], arr[i]
    return i

def kthSmallest(arr, l, r, k):

    if (k > 0 and k <= r - l + 1):
        index = partition(arr, l, r)
        if (index - l == k - 1):
            return arr[index]
        if (index - l > k - 1):
            return kthSmallest(arr, l, index - 1, k)
        return kthSmallest(arr, index + 1, r,
                            k - index + l - 1)
    return sys.maxsize

def knn(X, y, x, k):
    n = len(y)
    d = []
    for i in range(n):
        d.append(numpy.linalg.norm(x - X[i]))
    diff = d[:]
    kth = kthSmallest(diff, 0, n - 1, k)
    aggrigate = 0
    counter = 0
    for j in range(n):
        if (d[j] <= kth):
            aggrigate = aggrigate + y[j]
    return aggrigate / k

def ridgeRegression (X, y, l):
    n = len(y)
    d = 1
    if (isinstance(X[0], list)):
        d = len(X[0])
    one = numpy.ones(n)
    mins = (1/n)*(X.T.dot(y))
    mins = numpy.append(mins, (1/n)*(one.T.dot(y)))
    w1 = 2*l*numpy.identity(d) + (1/n)*(X.T @ X)
    w1extend = (1/n)*(X.T @ one)
    neww1 = numpy.c_[w1, w1extend]

    w2 = (1/n)*(one.T @ X)
    w2 = numpy.append(w2,1)
    
    A = numpy.vstack((neww1, w2))
    w = numpy.linalg.solve(A,mins)
    b = w[-1]
    w = w[:-1]
    return w, b
    
DEstimate = []
DEstimate.append([])
for t in X_test_D:
    DEstimate[0].append(knn(X_train_D, Y_train_D, t, 1))
DEstimate.append([])
for t in X_test_D:
    DEstimate[1].append(knn(X_train_D, Y_train_D, t, 2))
DEstimate.append([])
for t in X_test_D:
    DEstimate[2].append(knn(X_train_D, Y_train_D, t, 3))
DEstimate.append([])
for t in X_test_D:
    DEstimate[3].append(knn(X_train_D, Y_train_D, t, 4))
DEstimate.append([])
for t in X_test_D:
    DEstimate[4].append(knn(X_train_D, Y_train_D, t, 5))
DEstimate.append([])
for t in X_test_D:
    DEstimate[5].append(knn(X_train_D, Y_train_D, t, 6))
DEstimate.append([])
for t in X_test_D:
    DEstimate[6].append(knn(X_train_D, Y_train_D, t, 7))
DEstimate.append([])
for t in X_test_D:
    DEstimate[7].append(knn(X_train_D, Y_train_D, t, 8))
DEstimate.append([])
for t in X_test_D:
    DEstimate[8].append(knn(X_train_D, Y_train_D, t, 9))

wD, bD = ridgeRegression (X_train_D, Y_train_D, 0)
Dguess = numpy.multiply(X_test_D, wD) + bD
figD = plt.figure()
plt.title("A1 E4.2 Set D - 1")
plt.scatter(X_test_D, DEstimate[0], s=7, label='one')
plt.scatter(X_test_D, DEstimate[8], s=7, label='nine')
plt.scatter(X_test_D, Dguess, s=7, label='lr')
plt.legend(loc='upper right')
figD.savefig('a1q4bDPLOT1.png', facecolor='w', edgecolor='w')

Dmse = []
for i in range(0,9):
    Dmse.append(numpy.square(Y_test_D - DEstimate[i]).mean())

DLRerror = numpy.square(Y_test_D - Dguess).mean()


figMSED = plt.figure()
plt.title("A1 E4.2 Set D - 2")
plt.plot(range(1,10), Dmse, label = "k-NN")
plt.axhline(DLRerror, color = 'g', label = "LR")
plt.legend(loc='upper right')
figMSED.savefig('a1q4bDPLOT2.png', facecolor='w', edgecolor='w')

EEstimate = []
EEstimate.append([])
for t in X_test_E:
    EEstimate[0].append(knn(X_train_E, Y_train_E, t, 1))
EEstimate.append([])
for t in X_test_E:
    EEstimate[1].append(knn(X_train_E, Y_train_E, t, 2))
EEstimate.append([])
for t in X_test_E:
    EEstimate[2].append(knn(X_train_E, Y_train_E, t, 3))
EEstimate.append([])
for t in X_test_E:
    EEstimate[3].append(knn(X_train_E, Y_train_E, t, 4))
EEstimate.append([])
for t in X_test_E:
    EEstimate[4].append(knn(X_train_E, Y_train_E, t, 5))
EEstimate.append([])
for t in X_test_E:
    EEstimate[5].append(knn(X_train_E, Y_train_E, t, 6))
EEstimate.append([])
for t in X_test_E:
    EEstimate[6].append(knn(X_train_E, Y_train_E, t, 7))
EEstimate.append([])
for t in X_test_E:
    EEstimate[7].append(knn(X_train_E, Y_train_E, t, 8))
EEstimate.append([])
for t in X_test_E:
    EEstimate[8].append(knn(X_train_E, Y_train_E, t, 9))

wE, bE = ridgeRegression (X_train_E, Y_train_E, 0)
Eguess = numpy.multiply(X_test_E, wE) + bE
figE = plt.figure()
plt.title("A1 E4.2 Set E - 1")
plt.scatter(X_test_E, EEstimate[0], s=7, label='one')
plt.scatter(X_test_E, EEstimate[8], s=7, label='nine')
plt.scatter(X_test_E, Eguess, s=7, label='lr')
plt.legend(loc='upper right')
figE.savefig('a1q4bEPLOT1.png', facecolor='w', edgecolor='w')

Emse = []
for i in range(0,9):
    Emse.append(numpy.square(Y_test_E - EEstimate[i]).mean())

ELRerror = numpy.square(Y_test_E - Eguess).mean()


figMSEE = plt.figure()
plt.title("A1 E4.2 Set E - 2")
plt.plot(range(1,10), Emse, label = "k-NN")
plt.axhline(ELRerror, color = 'g', label = "LR")
plt.legend(loc='upper right')
figMSEE.savefig('a1q4bEPLOT2.png', facecolor='w', edgecolor='w')

FEstimate = []
FEstimate.append([])
for t in X_test_F:
    FEstimate[0].append(knn(X_train_F, Y_train_F, t, 1))
FEstimate.append([])
for t in X_test_F:
    FEstimate[1].append(knn(X_train_F, Y_train_F, t, 2))
FEstimate.append([])
for t in X_test_F:
    FEstimate[2].append(knn(X_train_F, Y_train_F, t, 3))
FEstimate.append([])
for t in X_test_F:
    FEstimate[3].append(knn(X_train_F, Y_train_F, t, 4))
FEstimate.append([])
for t in X_test_F:
    FEstimate[4].append(knn(X_train_F, Y_train_F, t, 5))
FEstimate.append([])
for t in X_test_F:
    FEstimate[5].append(knn(X_train_F, Y_train_F, t, 6))
FEstimate.append([])
for t in X_test_F:
    FEstimate[6].append(knn(X_train_F, Y_train_F, t, 7))
FEstimate.append([])
for t in X_test_F:
    FEstimate[7].append(knn(X_train_F, Y_train_F, t, 8))
FEstimate.append([])
for t in X_test_F:
    FEstimate[8].append(knn(X_train_F, Y_train_F, t, 9))

Fmse = []
for i in range(0,9):
    Fmse.append(numpy.square(Y_test_F - FEstimate[i]).mean())

wF, bF = ridgeRegression (X_train_F, Y_train_F, 0)
FGuess = numpy.multiply(X_test_F, wF) + bF

wF, bF = ridgeRegression (X_train_F, Y_train_F, 0)

FLRerror = 0
for ytest in Y_test_F:
    FLRerror = numpy.square(ytest - FGuess).mean()

figF = plt.figure()
plt.title("A1 E4.3")
plt.plot(range(1,10), Fmse, label = 'k-NN')
plt.axhline(FLRerror, color = 'g', label = "LR")
plt.legend(loc='upper right')
figF.savefig('a1q4c.png', facecolor='w', edgecolor='w')