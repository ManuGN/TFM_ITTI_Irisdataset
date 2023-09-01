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
with open('model.pkl','rb') as model_file:
    model = pickle.load(model_file)

# In[3]:

def run():

    image = Image.open('image.png')
    st.image(image,use_column_width=False)
    
    st.header("Predicción % Tipo Flor Iris:")
    
    st.write("Inserta los valores para saber la predicción del tipo de Flor de Iris")

    # Cuadro de mando para ingresar las características
    sepal-length = st.slider('Longitud del Sépalo (cm):', 2.0, 6.0)
    sepal-width = st.slider('Ancho del Sépalo (cm):', 0.0, 5.0)
    petal-length = st.slider('Longitud del Pétalo (cm):', 0.0, 3.0)
    petal-width = st.slider('Ancho del Pétalo (cm):', 0.0, 2.0)

    # Agregar un botón para realizar la predicción
    if st.button('Realizar Predicción'):
    # Realizar la predicción
    prediction = model.predict([[sepal-length, sepal-width, petal-length, petal-width]])[0]

    # Mapa de nombres de especies
    class_names = ['Setosa', 'Versicolor', 'Virginica']

    # Mostrar la especie predicha
    st.subheader('Especie de Iris Predicha:')
    st.write(class_names[prediction])
    

# In[4]:


if __name__ == '__main__':
    run() 

