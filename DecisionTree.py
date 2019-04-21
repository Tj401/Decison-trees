# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:16:13 2019

@author: kdandebo
"""

import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', 500)
df = pd.read_csv('C:/Users/kdandebo/Desktop/HomelatoptoKarthiklaptop/Python/datasetforpractice/iris.csv')
print(df.head(10))
x = df[['sepal_length', 'sepal_width' , 'petal_length' , 'petal_width']]
y = df['species']
#x.head(10)
from sklearn.model_selection import train_test_split
#import statsmodels.formula.api as smf
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
#train,test = train_test_split(df, test_size=0.3, random_state=101)
#from sklearn.linear_model import LinearRegression
print(x_train.shape)
print(x_train.head(10))

from sklearn.tree import DecisionTreeClassifier
DesTree = DecisionTreeClassifier()

# Train Decision Tree Classifer
DTree = DesTree.fit(x_train,y_train)

#3model1 = clf('species ~ sepal_length + sepal_width + petal_length + petal_width', data = train).fit()
#print(DTree.summary())

#3clf = clf.fit(train)

#Predict the response for test dataset
y_pred = DTree.predict(x_test)

print(y_pred)

from sklearn import metrics
accu = metrics.accuracy_score(y_test,y_pred)
print(accu)

#or, this is another way of finding accurancy
np.mean(y_test == y_pred)

#metrics.confusion_matrix(y_test,y_pred)