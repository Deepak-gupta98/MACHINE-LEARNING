# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:10:26 2020

@author: Deepak
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

chum = pd.read_csv(r".spyder-py3\chum data.csv")
chum.info()
chum.shape
chum.head()

y=chum["Churn"].value_counts()

import seaborn as sns

sns.barplot(y.index, y.values)

y.index
y.values
y
y_true = chum["Churn"][chum["Churn"]==True]

churn_per = (y_true.shape[0]/chum["Churn"].shape[0])*100
print("churn percentage: ", churn_per)
# 13.4 %

chum.describe()
chum.groupby(["state", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(10,5))

chum.groupby(["area code", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(10,5))

chum.groupby(["international plan", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(10,5))

chum.groupby(["voice mail plan", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(10,5))

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
chum["state"] = le.fit_transform(chum["state"])
chum.dtypes
chum.head()

y = chum['Churn'].to_numpy().astype(np.int)
y.size

chum.drop(["phone number","Churn"], axis=1, inplace=True)
chum.head()

X = chum.to_numpy().astype(np.float)
print(X)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, random_state=42)

#-------------------------------------------------------
# model of Logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
print("Accuracy of Logistic Regression is:", accuracy_score(y_test, y_pred))
print("confusion matrix is:\n", confusion_matrix(y_test, y_pred))

#   Accuracy of Logistic Regression is: 0.86810551558753
# confusion matrix is:
 #              [[350   7]
 #              [ 48  12]]

#------------------------------------------------------
# model of KNN
from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier(n_neighbors=2,metric='minkowski')
knc.fit(X_train,y_train)
y_pred=knc.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
print("Accuracy of KNN is:", accuracy_score(y_test, y_pred))
print("confusion matrix is:\n", confusion_matrix(y_test, y_pred))

# Accuracy of KNN is: 0.8776978417266187
# confusion matrix is:
#            [[356   1]
#            [ 50  10]]

#---------------------------------------------------------
# model of ensemble
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
print("Accuracy of RFC is:", accuracy_score(y_test, y_pred))
print("confusion matrix is:\n", confusion_matrix(y_test, y_pred))

# Accuracy of RFC is: 0.947242206235012
# confusion matrix is:
 #          [[355   2]
 #          [ 20  40]]

#------------------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
print("Accuracy of GBC is:", accuracy_score(y_test, y_pred))
print("confusion matrix is:\n", confusion_matrix(y_test, y_pred))

# Accuracy of GBC is: 0.9520383693045563
# confusion matrix is:
#                [[352   5]
#                [ 15  45]]

#---------------------------------------------------------
# model of svm
from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
print("Accuracy of SVC is:", accuracy_score(y_test, y_pred))
print("confusion matrix of SVC is:\n", confusion_matrix(y_test, y_pred))

# Accuracy of SVC is: 0.8992805755395683
# confusion matrix of SVC is:
#                    [[356   1]
#                    [ 41  19]]

#---------------------------------------------------------
feature_importance = gbc.feature_importances_
print (gbc.feature_importances_)

feat_importances = pd.Series(gbc.feature_importances_, index=chum.columns)
feat_importances = feat_importances.nlargest(19)

feat_importances.plot(kind='barh' , figsize=(10,10)) 