import numpy
import matplotlib.pyplot as plt
def plotGraph(data, xlabel, ylabel, title, filename, ymin=-1, ymax=-1):
	plt.plot(data)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	if ymin != -1:
		plt.ylim(ymin=ymin)
	if ymax != -1:
		plt.ylim(ymax=ymax)
	plt.savefig(filename)
	plt.clf()


def perceptron(X, y, w, b, max_pass = 500):
    mistake = [0]*max_pass
    n = len(y)
    for t in range(0, max_pass):
        mistake[t] = 0
        for i in range(0, n):
            if (y[i] * (numpy.dot(X[i], w) + b) <= 0):
                w = numpy.add(w, numpy.multiply(y[i],X[i]))
                b += y[i]
                mistake[t]+=1
    return w, b, mistake

spambase_X = numpy.matrix.transpose(numpy.genfromtxt('spambase_X.csv', delimiter=','))
spambase_y = numpy.genfromtxt('spambase_y.csv', delimiter=',')
w = numpy.zeros(len(spambase_X[0]))
b = 0
w,b,mistakes = perceptron(spambase_X, spambase_y, w, b, 500)
plt.plot(mistakes)
plt.title('Exercise 1: Perceptron Implementation')
plt.xlabel('Passes')
plt.ylabel('Mistakes')
plt.savefig('a1q1_perceptron.png', transparent=False)
