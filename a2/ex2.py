import numpy as np
import statsmodels.api as sm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#Exercise 2
#Usage: python3 ex2.py
#Load the files in some other way (e.g., pandas) if you prefer
X_train_A = np.loadtxt('data/X_train_A.csv', delimiter=",")
Y_train_A = np.loadtxt('data/Y_train_A.csv',  delimiter=",").astype(int)

X_train_B = np.loadtxt('data/X_train_B.csv', delimiter=",")
Y_train_B = np.loadtxt('data/Y_train_B.csv', delimiter=",").astype(int)
X_test_B = np.loadtxt('data/X_test_B.csv', delimiter=",")
Y_test_B = np.loadtxt('data/Y_test_B.csv', delimiter=",").astype(int)

# log_reg = sm.Logit(Y_train_A, X_train_A).fit()

# print(log_reg.summary())
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train_A, Y_train_A)