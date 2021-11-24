import matplotlib.pyplot as plt
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
accuracy_score(ytest, y_model)
confusion_matrix(ytest, y_model)
#print()
#print()
classification_report(ytest, y_model)

#Making dictionaries
uPlatform=data['Platform'].unique()
uPublisher=data['Publisher'].unique()
uGenre=data['Genre'].unique()
platdict={}
pubdict={}
genredict={}
for i in range(len(uPlatform)):
    platdict[uPlatform[i]]=i
for j in range(len(uPublisher)):
    pubdict[uPublisher[j]]=j
for k in range(len(uGenre)):
    genredict[uGenre[k]]=k
#RF.feature_importances_

#streamlit

st.header("Video Game Sales Analysis and Predictor")
#st.write(RF.feature_importances_)

option = st.sidebar.selectbox(
    'Select a mode',
     ['About','KNN Results','Model Prediction'])

if option=='About':
    st.header('About')
    st.write('Data pulled from vgchartz.com by user GregorySmith on Kaggle on video games with sales exceeding 100,000 copies.')
    st.write('Data has factors Year Released, Platform, Publisher, and genre, while the output is Global Sales Generated.')
    st.write('Data was analyzed using KNN model')
    #hart_data = pd.DataFrame(
    #p.random.randn(20, 3),
    #olumns=['a', 'b', 'c'])

    #t.line_chart(chart_data)

elif option=='KNN Results':
    
    st.header('KNN Results (n neighbours = 10)')
    st.write('Accuracy Score')
    st.write(accuracy_score(ytest, y_model))
    st.write('Confusion Matrix')
    st.write(confusion_matrix(ytest, y_model))
    st.write('Classification Report')
    st.write(classification_report(ytest, y_model))
    
    #p.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    #olumns=['lat', 'lon'])

    #t.map(map_data)

else:
    st.header('Model Prediction')
    st.write('Please insert appropriate values for prediction')
    st.write('Platform')
    platchoice=platdict[st.selectbox('Platform select',('PS','PS2','PS3','PS4','PSV','XB','PC','XOne','DC','GC','Wii','WiiU'))]
    st.write('Publisher')
    pubchoice=pubdict[st.selectbox('Publisher select',('Nintendo','Microsoft Game Studios','Sony Computer Entertainment','Activision'))]
    st.write('Genre')
    genchoice=genredict[st.selectbox('Genre select',('Sports', 'Platform', 'Racing', 'Role-Playing', 'Puzzle','Misc','Shooter','Simulation', 'Action', 'Fighting', 'Adventure', 'Strategy'))]
    st.write('Year Published')
    yearchoice=st.number_input('Pick a year', 1995, 2021)
    if st.button('Run!'):
        x={'Year':[yearchoice],'Platform_cat':[platchoice],'Publisher_cat':[pubchoice],'Genre_cat':[genchoice]
        xinput=pd.DataFrame(x)
        st.write(knn.predict(xinput)) 
    else:
        st.write('Have no predictions')
    
    
    
    

    #st.write('Before you continue, please read the [terms and conditions](https://www.gnu.org/licenses/gpl-3.0.en.html)')
    #show = st.checkbox('I agree the terms and conditions')
    #if show:
        #st.write(pd.DataFrame({
        #'Intplan': ['yes', 'yes', 'yes', 'no'],
        #'Churn Status': [0, 0, 0, 1]
        #}))


