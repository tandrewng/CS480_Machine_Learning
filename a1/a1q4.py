import numpy
import matplotlib.pyplot as plt

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
    return INT_MAX

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

estimate = []

for t in X_test_D:
    estimate.append(knn(X_train_D, Y_train_D, t, 1))

plt.plot(X_test_D, estimate, 'o')
plt.savefig('a1q4.png')