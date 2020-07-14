# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 20:51:10 2020

@author: Deepak
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel(r"C:\Users\Deepak\.spyder-py3\data_corona.xlsx")
data
data.info()
data.pop("Government Id",)
data.info()

data.fillna(data["Body_temp"].median(), inplace=True)
data.fillna(data["Age"].median(), inplace=True)

data.describe()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])
data["Infection_propb"] = le.fit_transform(data["Infection_propb"])

data.info()

y=data["Infection_propb"]
x=data.drop(["Infection_propb"], axis=1)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(x,y)

while(True):
    print("----------------CORONA VIRUS APPLICATION-----------------")
    print("")
    print("Select any choice as mention")
    print("1. Test  2. Exit")
    ch=int(input("Enter choice as mention above"))
    if(ch==1):
        bt = float(input("Enter Body Temp in Fer :"))
        age = float(input("Enter age in years :"))
        print("choice for gender :",{"Male":1,"Female":0})
        gen=int(input("Enter choice as given above for gender :"))
        print("choice for Entry :",{"Breathing issue":1,"no issue":0})
        brp=int(input("Enter choice of Breathing Problem :"))
        print("Enter choice :",{"noise issue":1,"no issue":0})
        rn=int(input("Enter choice for running nose :"))
        print("choice for entry :",{"Body pain issue":1,"no issue":0})
        bp=int(input("Enter choice for body pain issue :"))
        
        x_test=[bt,age,gen,brp,rn,bp]
        ref={1:"Test is positive", 0: "Test is Negative"}
        y_pred=model.predict([x_test,])
        print("the report of the patient is ")
        print(ref[y_pred[0]])
        
    elif(ch == 2):
        print("Do you want to exit")
        break
    else :
        print("Enter choice as mention")