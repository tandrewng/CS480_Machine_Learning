# run ./q2.sh
# target files have prefix a1q2 and file type png
import numpy
import matplotlib.pyplot as plt

housing_X_test = numpy.matrix.transpose(numpy.genfromtxt('housing_X_test.csv', delimiter=','))
housing_X_train = numpy.matrix.transpose(numpy.genfromtxt('housing_X_train.csv', delimiter=','))

housing_y_test = numpy.genfromtxt('housing_y_test.csv', delimiter=',')
housing_y_train = numpy.genfromtxt('housing_y_train.csv', delimiter=',')

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
    #b1 = 1/n*X.T
    #b2 = 1/n*one.T
    
    A = numpy.vstack((neww1, w2))
    w = numpy.linalg.solve(A,mins)
    b = w[-1]
    w = w[:-1]
    return w, b

def gradientDescent (X, y, l, max_pass, step, tol):
    n = len(y)
    d = len(X[0])
    wt = [numpy.zeros(d)]
    bt = [0]
    ones = numpy.ones(n)
    for i in range(1, max_pass+1):
        wt.append(wt[-1] - step*((1/n)*((1/n)*(X.T @ (X @ wt[-1] + bt[-1] - y))) + 2 * wt[-1].dot(l)))
        bt.append(bt[-1] - step*((1/n)*((1/n)* (ones.T @ (X @ wt[-2] + bt[-1] - y)))))
        if(numpy.linalg.norm(wt[-1]-wt[-2]) <= tol):
            break

    return wt, bt

n = len(housing_X_test)

# RR lambda = 0
wRR0, bRR0 = ridgeRegression(housing_X_train, housing_y_train, 0)

trainerrorRR0 = numpy.sum(numpy.square(housing_X_train.dot(wRR0) + bRR0 - housing_y_train))/(2 * n)
print("Training error for RR with lambda = 0 is",trainerrorRR0)

trainlossRR0 = trainerrorRR0 + 0 * numpy.sum(numpy.square(wRR0))
print("Training loss for RR with lambda = 0 is", trainlossRR0)

testerrorRR0 = numpy.sum(numpy.square(housing_X_test.dot(wRR0) + bRR0 - housing_y_test))/(2 * n)
print("Testing error for RR with lambda = 0 is", testerrorRR0)

# RR lambda = 10
wRR10, bRR10 = ridgeRegression(housing_X_train, housing_y_train, 10)

trainerrorRR10 = numpy.sum(numpy.square(housing_X_train.dot(wRR10) + bRR10 - housing_y_train))/(2 * n)
print("Training error for RR with lambda = 10 is",trainerrorRR10)

trainlossRR10 = trainerrorRR10 + 0 * numpy.sum(numpy.square(wRR10))
print("Training loss for RR with lambda = 10 is", trainlossRR10)

testerrorRR10 = numpy.sum(numpy.square(housing_X_test.dot(wRR10) + bRR10 - housing_y_test))/(2 * n)
print("Testing error for RR with lambda = 10 is", testerrorRR10)

# GD lambda = 0
wGD0, bGD0 = gradientDescent(housing_X_train, housing_y_train, 0, 100000, 0.00001, 0.0000001)

trainerrorGD0 = numpy.sum(numpy.square(housing_X_train.dot(wGD0[-1]) + bGD0[-1] - housing_y_train))/(2 * n)
print("Training error for GD with lambda = 0 is", trainerrorGD0)

trainlossGD0 = trainerrorGD0 + 0 * numpy.sum(numpy.square(wGD0[-1]))
print("Training loss for GD with lambda = 0 is", trainlossGD0)

testerrorGD0 = numpy.sum(numpy.square(housing_X_test.dot(wGD0[-1]) + bGD0[-1] - housing_y_test))/(2 * n)
print("Testing error for GD with lambda = 0 is", testerrorGD0)


# RR lambda = 10
wGD10, bGD10 = gradientDescent(housing_X_train, housing_y_train, 10, 100000, 0.00001, 0.0000001)

trainerrorGD10 = numpy.sum(numpy.square(housing_X_train.dot(wGD10[-1]) + bGD10[-1] - housing_y_train))/(2 * n)
print("Training error for GD with lambda = 10 is", trainerrorGD10)

trainlossGD10 = trainerrorGD10 + 0 * numpy.sum(numpy.square(wGD10[-1]))
print("Training loss for GD with lambda = 10 is", trainlossGD10)

testerrorGD10 = numpy.sum(numpy.square(housing_X_test.dot(wGD10[-1]) + bGD10[-1] - housing_y_test))/(2 * n)
print("Testing error for GD with lambda = 10 is", testerrorGD10)

traininglossiterGD0 = []
traininglossiterGD10 = []
for i in range(1000000):
    if i < len(wGD0):
        traininglossiterGD0.append(numpy.sum(numpy.square(housing_X_train.dot(wGD0[i]) + bGD0[i] - housing_y_train))/(2 * n) + 0 * numpy.sum(numpy.square(wGD0[i])))

    if i < len(wGD10):
        traininglossiterGD10.append(numpy.sum(numpy.square(housing_X_train.dot(wGD10[i]) + bGD10[i] - housing_y_train))/(2 * n) + 0 * numpy.sum(numpy.square(wGD10[i])))

plt.title("A1 E2")
plt.plot(range(len(wGD0)), traininglossiterGD0, label = 'GD lambda = 0')
plt.plot(range(len(wGD10)), traininglossiterGD10, label = 'GD lambda = 10')
plt.legend(loc='upper right')
plt.savefig('a1q2e.png', facecolor='w', edgecolor='w')