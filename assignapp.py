!pip install streamlit
!pip install pyngrok
!pip install dtale

import seaborn as sns
import pandas as pd
import numpy as np
import dtale
import streamlit as st
import sklearn

data=pd.read_csv('vgsales.csv')
data.isnull().sum()

#Dropping null values
data.dropna(subset=['Year', 'Publisher'], inplace=True)
data.isnull().sum()

data.drop(['Name','Rank'], axis=1)

#Encoding str values
data['Platform']=data['Platform'].astype('category')
data['Platform_cat']=data['Platform'].cat.codes
data['Publisher']=data['Publisher'].astype('category')
data['Publisher_cat']=data['Publisher'].cat.codes
data['Genre']=data['Genre'].astype('category')
data['Genre_cat']=data['Genre'].cat.codes
data.head(6)

#Data split
X = data.drop(['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Platform', 'Genre', 'Publisher','Name','Rank'], axis = 1)
y = data['Global_Sales'].astype(int) #change from float to int because KNN can't handle y as float
#X.head()

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state = 99)

## KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 10)

knn.fit(Xtrain, ytrain)

print(knn.score(Xtest, ytest))

from sklearn.metrics import accuracy_score

y_model = knn.predict(Xtest) 
print(accuracy_score(ytest, y_model))

## Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

logreg = LogisticRegression()
logreg.fit(Xtrain, ytrain)
ypred = logreg.predict(Xtest)

print(confusion_matrix(ytest, ypred))
print()
print()
print(classification_report(ytest, ypred))

## Support Vector Machine Classifier
from sklearn.svm import SVC

svc = SVC()
svc.fit(Xtrain, ytrain)
ypred = svc.predict(Xtest)

print(confusion_matrix(ytest, ypred))
print()
print()
print(classification_report(ytest, ypred))

## Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(Xtrain, ytrain)
ypred = nb.predict(Xtest)

print(confusion_matrix(ytest, ypred))
print()
print()
print(classification_report(ytest, ypred))

## Random Forest
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier()
RF.fit(Xtrain, ytrain)
ypred = RF.predict(Xtest)

print(confusion_matrix(ytest, ypred))
print()
print()
print(classification_report(ytest, ypred))

RF.feature_importances_

#streamlit

st.header("My first Streamlit App")
st.write(RF.feature_importances_)
!streamlit run myfirstapp.py --server.port 80 &>/dev/null&
from pyngrok import ngrok
ngrok.kill()

public_url = ngrok.connect(port = 80)
public_url
