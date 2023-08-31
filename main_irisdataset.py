#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import pickle 
from pickle import dump
from PIL import Image
import streamlit.components.v1 as components


# In[2]:


with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# In[3]:


def run():

    image = Image.open('image.png')
    st.image(image,use_column_width=False)
    
    st.header("Predicción % Tipo Flor Iris:")
    
    st.write("Inserta los valores para saber la predicción del tipo de Flor de Iris")
    
    SepalLengthCm = st.slider('SepalLengthCm:', 2.0, 6.0)
    SepalWidthCm = st.slider('SepalWidthCm:', 0.0, 5.0)
    PetalLengthCm = st.slider('PetalLengthCm',0.0, 3.0)
    PetalWidthCm = st.slider('SkiPetalWidthCm:', 0.0, 2.0)
    
    data = {'SepalLengthCm': SepalLengthCm,
            'SepalWidthCm': SepalWidthCm,
            'PetalLengthCm': PetalLengthCm,
            'PetalWidthCm': PetalWidthCm}
    
    features = pd.DataFrame(data, index=[0])
    
    pred_proba = model.predict_proba(features)
    
    st.subheader('Prediccion en Porcentajes:')
    
    st.write('**Probablity of Iris Class being Iris-setosa is ( in % )**:',pred_proba[0][0]*100)
    st.write('**Probablity of Isis Class being Iris-versicolor is ( in % )**:',pred_proba[0][1]*100)
    st.write('**Probablity of Isis Class being Iris-virginica ( in % )**:',pred_proba[0][2]*100)


# In[4]:


if __name__ == '__main__':
    run() 

