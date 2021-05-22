# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 13:49:03 2020

@author: NIKHIL GUPTA
"""

import numpy as np
import pandas as pd

dataset_train = pd.read_csv('Train.csv')
dataset_train.fillna('0',inplace = True)

#Make INCIDENT_ID as index and drop date Column in train dataset
dataset_train['DATE'] = pd.to_datetime(dataset_train['DATE'])
dataset_train['day'] = dataset_train['DATE'].dt.day
dataset_train['month'] = dataset_train['DATE'].dt.month

dataset_train = dataset_train.drop(['DATE'],axis = 1)
dataset_train = dataset_train.set_index('INCIDENT_ID')
target = 'MULTIPLE_OFFENSE'
X = dataset_train.iloc[:,dataset_train.columns!=target].values
Y = dataset_train.iloc[:,-3].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Make INCIDENT_ID as index and drop date Column in test dataset
dataset_test = pd.read_csv('Test.csv')
dataset_test.fillna('0',inplace = True)
dataset_test['DATE'] = pd.to_datetime(dataset_test['DATE'])
dataset_test['day'] = dataset_test['DATE'].dt.day
dataset_test['month'] = dataset_test['DATE'].dt.month

dataset_test = dataset_test.drop(['DATE'],axis=1)
dataset_test = dataset_test.set_index('INCIDENT_ID')
Xtest = dataset_test.iloc[:,0:19].values

#XGBoost Algorithm
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, Y_train)

#predictions for train data
Y_pred = classifier.predict(X_test)

#Accuracy, F score and confusion matix 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
cm_train = confusion_matrix(Y_test, Y_pred)
acc_train = accuracy_score(Y_test,Y_pred)
fscore_train = f1_score(Y_test,Y_pred)

#Predictions for test data
Y_pred2 = classifier.predict(Xtest)

#Writing the prediction in csv file
Ywrite=pd.DataFrame(Y_pred2,columns=['MULTIPLE_OFFENSE'])
var =pd.DataFrame(dataset_test.index)
dataset_test_col = pd.concat([var,Ywrite], axis=1)
dataset_test_col.to_csv("Prediction.csv",index=False)


