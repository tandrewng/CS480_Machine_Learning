import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import csv

x_train = np.array(pd.read_csv('./train_x.csv', header=0))
xlength, xwidth = x_train.shape
x_train = x_train[:, [i for i in range(1, xwidth)]]

x_test = np.array(pd.read_csv('./test_x.csv', header=0))
x_test = x_test[:, [i for i in range(1, xwidth)]]

y_train = np.array(pd.read_csv('./train_y.csv', header=0))
ylen, ywidth = y_train.shape 
y_train = y_train[:, [i for i in range(1, ywidth)]]

dtc = RandomForestClassifier(criterion='entropy', max_depth=17, max_features="sqrt", min_samples_split=17)
bc = BaggingClassifier(base_estimator=dtc, n_estimators=70)
bc.fit(x_train, y_train)

y_predict = bc.predict(x_test)

f = open('./foo1.csv', 'w', newline ='')
writer = csv.writer(f)
writer.writerow(["index","label"])

for i in range(len(y_predict)):
    writer.writerow([i, y_predict[i]])

f.close()