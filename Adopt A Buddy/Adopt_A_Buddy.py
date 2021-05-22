# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:46:26 2020

@author: NIKHIL GUPTA
"""

import numpy as np
import pandas as pd

dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')

dataset_train.fillna('0',inplace = True)
dataset_test.fillna('0', inplace = True)


dataset_train = dataset_train.drop(['issue_date','listing_date'], axis = 1)
dataset_train = dataset_train.set_index('pet_id')

dataset_test = dataset_test.drop(['issue_date','listing_date'], axis = 1)
dataset_test = dataset_test.set_index('pet_id')

#Pet Type
Xtrain_type = dataset_train[['condition','length(m)','height(cm)','X1','X2']]
Ytrain_type = dataset_train.iloc[:,-1].values
Ytrain_breed = dataset_train.iloc[:,-2].values

Xtrain_type = Xtrain_type.astype(float)

from sklearn.model_selection import train_test_split
X_traintype, X_testtype, Y_traintype, Y_testtype = train_test_split(Xtrain_type, Ytrain_type, test_size = 0.2, random_state = 42)

X_trainbreed, X_testbreed, Y_trainbreed, Y_testbreed = train_test_split(Xtrain_type, Ytrain_breed, test_size = 0.2, random_state = 42)

from xgboost import XGBClassifier
classifier_type = XGBClassifier()
classifier_type.fit(X_traintype, Y_traintype)

classifier_breed = XGBClassifier()
classifier_breed.fit(X_trainbreed, Y_trainbreed)

Xtest = dataset_train[['condition','length(m)','height(cm)','X1','X2']]
Xtest = Xtest.astype(float)

Y_pred_type = classifier_type.predict(Xtest)
Y_pred_breed = classifier_breed.predict(Xtest)

"""
cm_type = confusion_matrix(Xtest, Y_pred_type)
acc_type = accuracy_score(Y_testbreed, Y_pred_type)

cm_breed = confusion_matrix(Xtest, Y_pred_breed)
acc_breed = accuracy_score(Y_testbreed, Y_pred_breed)
"""

Ywrite_type = pd.DataFrame(Y_pred_type, columns = ['pet_category'])
Ywrite_breed = pd.DataFrame(Y_pred_breed, columns = ['breed_category']) 
var =pd.DataFrame(dataset_test.index)
dataset_test_col = pd.concat([var,Ywrite_breed,Ywrite_type], axis=1)
dataset_test_col.to_csv("Prediction.csv", index=False)