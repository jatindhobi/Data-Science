import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle as pk 


df=pd.read_csv(r"C:\5th Sem\Data-Science\Datasets\heart_failure_clinical_records_dataset.csv")
df

x=df[["age"]]
y=df[["platelets"]]

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=50)
model=LinearRegression()

model.fit(x_train,y_train)

model.predict([[50.0]])