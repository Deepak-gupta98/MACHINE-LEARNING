# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 09:13:41 2020

@author: Deepak
"""


from tkinter import*
from sklearn.datasets import load_iris
import pymysql
import pyttsx3
from tkinter import messagebox

root=Tk()
root.geometry("400x600")
root.configure(background="light green")
root.title("Iris Dataset")

Label(root,text="GUI BASED FLOWER CLASSIFIER ",font=('Helvetica',15,'bold'),bg="light green", relief="solid").pack()

Label(root,text="application version 1.1 ",relief="solid",bg="light green").pack(side=BOTTOM)

Label(root,text="============================================", bg="light green").pack()

Label(root,text="Sepal length in cm ",font=('Helvetica',10,'bold'),bg="light green", relief="solid", width=18).place(x=40,y=80)

Label(root,text="Sepal width in cm ",font=('Helvetica',10,'bold'),bg="light green", relief="solid", width=18).place(x=40,y=130)

Label(root,text="Petal length in cm ",font=('Helvetica',10,'bold'),bg="light green", relief="solid", width=18).place(x=40, y= 180)

Label(root,text="Petal width in cm ",font=('Helvetica',10,'bold'),bg="light green", relief="solid", width=18).place(x=40,y=230)

Label(root,text="Information ", font=('Helvetica',10,'bold'), bg="light green", relief="solid",width=18).place(x=40,y=280)

Label(root,text="Prediction Result is ",font=('Helvetica',10,'bold'),bg="light green", relief="solid", width=18).place(x=40,y=320)

sl=StringVar()
sw=StringVar()
pl=StringVar()
pw=StringVar()

Entry(root,text=sl,width=25).place(x=200,y=80)
Entry(root,text=sw,width=25).place(x=200,y=130)
Entry(root,text=pl,width=25).place(x=200,y=180)
Entry(root,text=pw,width=25).place(x=200,y=230)


def info():
    n=Tk()
    n.geometry("500x500")
    n.configure(background="white")
    
    Label(n,text="Information About Application",bg="light green",font=('Helvetica',10,'bold')).pack()                      ,  
    
    data='''The Application is a Machine learning GUI Based Product Developed by Deepak Kumar .
    The Product is iris Flower classifier and the model classify three species of flower that are know as [Iris-Setosa,Iris-Versicolor,Iris-Virginica]
    
    To Predict the class you have to enter certain parameter such as 
    Petal length,Petal width, sepal length, Sepal width in cm.
    
    Data set Characteristics:**
    
    :Number of instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric,predictive attributes and the class
    :Attribute Information:
        -sepal length in cm
        -sepal width in cm
        -petal length in cm
        -petal width in cm
        -class:
            -Iris-Setosa
            -Iris-Versicolor
            -Iris-Virginica
          '''
            
    Label(n,text=data).place(x=30,y=30)
    n.resizable(0,0)
    n.mainloop()
    
Button(root,text="info",width=18,command=info).place(x=200,y=279)

def model():
    global conn
    conn = pymysql.connect('localhost','root','','machine_learning')
    cur = conn.cursor()
    data = load_iris()
    x = data.data
    y = data.target
    
    from sklearn.neighbors import KNeighborsClassifier
    model=KNeighborsClassifier(n_neighbors=5,metric='euclidean')
    model.fit(x,y)
    x_test=[float(sl.get()),float(sw.get()),float(pl.get()),float(pw.get())]
    y_pred=model.predict([x_test,])
    
    Label(root,text=str(data.target_names[y_pred[0]]),font=('Helvetica',10,'bold'),bg="light green",relief="solid",width=18).place(x=200,y=320)
    
    engine=pyttsx3.init()
    voices=engine.getProperty('voices')
    rate=engine.getProperty('rate')
    engine.setProperty('rate',rate-100)
    engine.say('''Welcome to our Flower Classification system the predicted species is'''+str(data.target_names[y_pred[0]]))
    engine.runAndWait()

    q="insert into iris values(%s,%s,%s,%s,%s)"   
    val=(float(sl.get()), float(sw.get()), float(pl.get()),float(pw.get()),str(data.target_names[y_pred[0]]))
    
    cur.execute(q,val)
    conn.commit()
     
    ans=Tk()
    label(ans,text=str(data.target_names[y_pred[0]]),font=('Helvetica',10,'bold'),bg="light green", relief="solid", width=18).pack()
    ans.mainloop()
    
    messagebox.showinfo("answer",str(data.target_names[y_pred[0]]))
    
Button(root,text="Prediction",width=18,command=model).place(x=40,y=400)

def destroy():
    conn.close()
    root.destroy()
    
Button(root,text="Termination",width=18,command=destroy).place(x=200,y=400)


def clear():
    Label(root,text=" "*30,font=('Helvetica',10,'bold'),bg="light green",relief="solid", width=18).place(x=200,y=320)
    
Button(root,text="Clear Prediction", width=18, command=clear).place(x=110,y=350)
    
root.resizable(0,0)
root.mainloop()
    
    
    
    