import pandas as pd
import numpy as np
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

knn = KNeighborsClassifier(n_neighbors = 10)

knn.fit(Xtrain, ytrain)

#print(knn.score(Xtest, ytest))

y_model = knn.predict(Xtest) 
#print(accuracy_score(ytest, y_model))
#print(confusion_matrix(ytest, ypred))
#print()
#print()
#print(classification_report(ytest, ypred))

RF.feature_importances_

#streamlit

st.header("Video Game Sales Analysis and Predictor")
st.write(RF.feature_importances_)

st.header("My first Streamlit App")

option = st.sidebar.selectbox(
    'Select a mini project',
     ['line chart','map','T n C'])

if option=='line chart':
    chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])

    st.line_chart(chart_data)

elif option=='map':
    map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

    st.map(map_data)

else:

    st.write('Before you continue, please read the [terms and conditions](https://www.gnu.org/licenses/gpl-3.0.en.html)')
    show = st.checkbox('I agree the terms and conditions')
    if show:
        st.write(pd.DataFrame({
        'Intplan': ['yes', 'yes', 'yes', 'no'],
        'Churn Status': [0, 0, 0, 1]
        }))


