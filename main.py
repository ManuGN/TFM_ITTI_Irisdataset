#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import streamlit as st
import pickle
from PIL import Image


# In[2]:


# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target


# In[3]:


# Train a machine learning model
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)


# In[4]:


filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))


# In[5]:


image = Image.open('image.png')
st.image(image,use_column_width=False)


# In[6]:


K = False


# In[7]:


filename = 'model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))


# In[8]:


l = ["Setosa", "Verginica", "Versicolor"]


# In[9]:


def predict(x):
    return loaded_model.predict(x)


# In[10]:


st.title('IRIS Prediction App')


# In[11]:


# Add a text input widget for the user to enter data
x_input = st.text_input('Dame los siguientes valores: Petal Length, Petal Width, Sepal Length and Sepal Width:')


# In[12]:


# Cuando el usuario pulse el botón Predicción, se hará una predicción del tipo de Iris según los datos cargados:
if st.button('Prediccion'):
    x = [float(l) for l in x_input.split()]
    y_pred = predict([x])
    st.write('El tipo de especie es: ', l[y_pred[0]])
    K = True


# In[13]:


# Create a container for the image and a text widget
container = st.container()


# In[14]:


# Create two columns inside the container
col1, col2 = container.columns([1, 4])


# In[15]:


if K == True:
    with col2:
        st.image(image, use_column_width=True)

