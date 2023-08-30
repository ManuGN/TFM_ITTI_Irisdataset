#!/usr/bin/env python
# coding: utf-8

# In[1]:
import Sklearn
import streamlit as st
import pandas as pd
import pickle
from pickle import dump
from PIL import Image

# In[2]:

Iris = pd.read_csv('Iris.csv')
species = ['setosa', 'versicolor', 'virginica']
image = ['setosa.jpg', 'versicolor.jpg', 'virginica.jpg']
with open('App_Iris.pkl', 'rb') as f:
     App_Iris = pickle.load(f)

# In[3]:

image = Image.open('image.png')
st.image(image,use_column_width=False)


# In[4]:


st.header("Tipo Flor Iris:")


# In[5]:


st.write("Inserta los valores para saber la predicción del tipo de Flor de Iris")


# In[6]:


SepalLengthCm = st.slider('SepalLengthCm:', 2.0, 6.0)
SepalWidthCm = st.slider('SepalWidthCm:', 0.0, 5.0)
PetalLengthCm = st.slider('PetalLengthCm',0.0, 3.0)
PetalWidthCm = st.slider('SkiPetalWidthCm:', 0.0, 2.0)


# In[7]:


Iris = {'SepalLengthCm': SepalLengthCm,
        'SepalWidthCm': SepalWidthCm,
        'PetalLengthCm': PetalLengthCm,
        'PetalWidthCm': PetalWidthCm}


# In[8]:


features = pd.DataFrame(data, index=[0])


# In[9]:


pred_proba = model.predict_proba(features)


# In[10]:


st.subheader('Predicción porcentaje TIPO DE IRIS:')


# In[11]:


st.write('**Probablity of Iris Class being Iris-setosa is ( in % )**:',pred_proba[0][0]*100)
st.write('**Probablity of Isis Class being Iris-versicolor is ( in % )**:',pred_proba[0][1]*100)
st.write('**Probablity of Isis Class being Iris-virginica ( in % )**:',pred_proba[0][2]*100)

