
import pandas as pd
import numpy as np

import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt


dataset = pd.read_csv("SaYoPillow.csv")
a = dataset.drop("stress level",axis=1)
b = dataset["stress level"]
a_train,a_test,b_train,b_test = train_test_split(a,b,test_size=0.2)
modelb = SVC(kernel='linear').fit(a_train,b_train)
b_pred = modelb.predict(a_test)

data13=pd.read_csv("heart.csv")
data13=data13.drop(["exng","oldpeak","slp","caa","thall","chol"],axis=1)
x=data13.drop("output",axis=1)
y=data13["output"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
np.random.seed(42)
model = LogisticRegression().fit(x_train,y_train)
y_pred = model.predict(x_test)
acc=accuracy_score(y_pred,y_test)



nav = st.sidebar.radio("Navigation",["Stress Calculator","Heart Attack Pridiction"])
if nav == "Stress Calculator":
    st.title("Stress Calculator!!!!")
    int1 = st.number_input("Please Enter Snoring rate")
    int2 = st.number_input("Please Enter Respiration rate")
    int3 = st.number_input("Please Enter Body temperature")
    int4 = st.number_input("Please Enter Limb movement")
    int5 = st.number_input("Please Enter Blood oxygen")
    int6 = st.number_input("Please Enter Eye movement")
    int7 = st.number_input("Please Enter Sleeping hours")
    int8 = st.number_input("Please Enter Heart rate")
    st.button("Submit")
    input_data = (int1,int2,int3,int4,int5,int6,int7,int8)
    input_data = np.asarray(input_data)
    input_data = input_data.reshape(1, -1)
    pred = modelb.predict(input_data)

    if pred[0] == 0:
        st.success(f"Well Done!!! You Have No Stress and You get score : {pred[0]}")
    elif pred[0] == 1:
        st.success(f"Well Done!!! You Have Little bit Stress and You get score: {pred[0]}")
    elif pred[0] == 2:
        st.success(f" You Have little bit Stress and You get score : {pred[0]}")
    elif pred[0] == 3:
        st.success(f" You Have Little bit more Stress and You get score : {pred[0]}")
    elif pred[0] == 4:
        st.success(f"It's Dengerous!!! You Have to much Stress and You get score : {pred[0]}")
if nav == "Heart Attack Pridiction":
    st.title("Heart Attack Pridiction")
    Age = st.number_input("Please Enter Age")
    Sex = st.radio("Sex", ["Male", "Female"])
    if Sex == "Male":
        Sex = 1
    if Sex == "Female":
        Sex = 0
    Chest_pain = st.number_input("Please Enter Chest pain Ratio")
    BP = st.number_input("Please Enter Blood pressure")
    Cholestroale = st.number_input("Please Enter Cholestrole")
    Sugar = st.radio("Sugar", ["Yes", "No"])
    if Sugar == "Yes":
        Sugar = 1
    if Sugar == "No":
        Sugar = 0
    Heart_rate= st.number_input("Please Enter Heart-Rate")
    st.button("Submit")
    input_data1 = (Age,Sex,Chest_pain,BP,Cholestroale,Sugar,Heart_rate)
    input_data1 = np.asarray(input_data1)
    input_data1 = input_data1.reshape(1, -1)
    pred1 = model.predict(input_data1)
    if pred1[0] == 0:
        st.success(f"Well Done!!! You Have No Heart Disease And Our Model Accuracy is : {acc*100}")
    elif pred1[0] == 1:
        st.success(f"You Have Little bit Heart Disease And Our Model Accuracy is : {acc*100}")



