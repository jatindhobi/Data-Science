import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import pickle
import streamlit as st

df=pd.read_csv(r"C:\5th Sem\Data-Science\Datasets\Iris.csv")
x=df.drop(["Species","Id"],axis=1)
y=df["Species"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

pickle_out=open("classifier.pkl","wb")
pickle.dump(knn,pickle_out)
pickle_out.close()

st.markdown('## Iris Species Prediction')
Sepal_length=st.number_input("Sepal Length(cm)")
Sepal_width=st.number_input("Sepal Width (cm)")
petal_length=st.number_input("petal Length(Cm)")
petal_width=st.number_input("Petal Width (Cm)")
if st.button('Predict'):
    model=joblib.load("classifier.pkl")
    x=np.array([Sepal_length,Sepal_width,petal_length,petal_width])
    if any(x<=0):
        st.markdown("### Input must be greater than 0")
    else:
        st.markdown(f'### Prediction Is {model.predict([[Sepal_length,Sepal_width,petal_length,petal_width]])}')