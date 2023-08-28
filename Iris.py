#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from sklearn.ensemble import RandomForestClassifier


# In[2]:


model = pickle.load(open('App_Iris.pkl', 'rb'))


# In[3]:


st.header("Iris Classification:")


# In[4]:


image = Image.open('image.png')
st.image(image,use_column_width=False)


# In[5]:


st.write("Please insert values, to get Iris class prediction")


# In[6]:


SepalLengthCm = st.slider('SepalLengthCm:', 2.0, 6.0)
SepalWidthCm = st.slider('SepalWidthCm:', 0.0, 5.0)
PetalLengthCm = st.slider('PetalLengthCm',0.0, 3.0)
PetalWidthCm = st.slider('SkiPetalWidthCm:', 0.0, 2.0)


# In[7]:


data = {'SepalLengthCm': SepalLengthCm,
        'SepalWidthCm': SepalWidthCm,
        'PetalLengthCm': PetalLengthCm,
        'PetalWidthCm': PetalWidthCm}


# In[8]:


features = pd.DataFrame(data, index=[0])


# In[9]:


pred_proba = model.predict_proba(features)


# In[10]:


st.subheader('Prediction Percentages:')


# In[11]:


st.write('**Probablity of Iris Class being Iris-setosa is ( in % )**:',pred_proba[0][0]*100)
st.write('**Probablity of Isis Class being Iris-versicolor is ( in % )**:',pred_proba[0][1]*100)
st.write('**Probablity of Isis Class being Iris-virginica ( in % )**:',pred_proba[0][2]*100)

