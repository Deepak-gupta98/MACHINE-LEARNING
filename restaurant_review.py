# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 22:36:05 2020

@author: Deepak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"C:\Users\Deepak\.spyder-py3\Restaurant_Reviews.tsv",delimiter='/t')

import re  
import nltk  
  
nltk.download('stopwords') 
  
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
  
corpus = []  
  
for i in range(0, 1000):  
      
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  
    review = review.lower()  
    
    review = review.split()  
      
    ps = PorterStemmer()  
        
    review = [ps.stem(word) for word in review 
                if not word in set(stopwords.words('english'))]  
                  
    review = ' '.join(review)   
    corpus.append(review)  

# count the sentence
from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer(max_features = 1500)  
X = cv.fit_transform(corpus).toarray()  
y = dataset.iloc[:, 0].values 

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 


from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(n_estimators = 501, 
                            criterion = 'entropy')
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) 

# check Confusion matrix
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred) 
  
