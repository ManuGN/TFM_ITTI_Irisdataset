import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import streamlit as st
import pickle

# Leer los datos
df = pd.read_csv('creditcard.csv')
df.head()

#Luego, se aplican las técnicas de preprocesamiento y limpieza de datos:
# eliminamos la columna "Time"
df = df.drop('Time', axis=1)

# crear una columna adicional para indicar si la transacción es fraudulenta o no
df['fraud'] = df['Class'].apply(lambda x: 'Fraud' if x == 1 else 'Not Fraud')


# Crear la aplicación Streamlit

st.title("**Tarjetas de Crédito**")

# Cuadro de mando con parámetros de los datos
st.subheader("Datos Importe / Fraude")
total_transactions = df.shape[0]
fraud_transactions = df[df["fraud"] == "Fraud"].shape[0]
not_fraud_transactions = df[df["fraud"] == "Not Fraud"].shape[0]
st.write(f"Total de transacciones: {total_transactions}")

    # Crear una tabla interactiva con los parámetros Amount y Fraud
parameter_table = pd.DataFrame({
        'Amount': df['Amount'],
        'Fraud': df['fraud']
    })
st.dataframe(parameter_table)

# Botón para ver la cantidad de transacciones con fraude y no fraude
if st.button('Botón total de transacciones Fraudulentas / No fraudulentas'):
        fraud_transactions = df[df['fraud'] == 'Fraud'].shape[0]
        not_fraud_transactions = df[df['fraud'] == 'Not Fraud'].shape[0]
        st.write(f"Transacciones fraudulentas: {fraud_transactions}")
        st.write(f"Transacciones no fraudulentas: {not_fraud_transactions}")

st.markdown("**----------------------------------------------------------**")

# Gráfico de barras de las transacciones fraudulentas y no fraudulentas
fig, ax = plt.subplots()
counts = [fraud_transactions, not_fraud_transactions]
labels = ["Fraud", "Not Fraud"]
ax.bar(labels, counts)
ax.set_xlabel("Tipo de transacción")
ax.set_ylabel("Cantidad de transacciones")
ax.set_title("Datos no balanceados")
st.pyplot(fig)

st.markdown("**----------------------------------------------------------**")

# Cargar el conjunto de datos
data = pd.read_csv("creditcard.csv")

# Seleccionar las variables de interés - Las Variables V1,V2,V3.....V28 son datos no relevantes porque son anonimizados.
selected_features = ["Time", "Amount"]
X = data[selected_features]
y = data["Class"]

# Aplicar SMOTE para abordar el desbalanceo de datos
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Entrenar el modelo de Random Forest con las características seleccionadas
model = RandomForestClassifier()
model.fit(X_res, y_res)

# Crear el cuadro de mando con Streamlit
st.title("Cuadro de Mando - Detección de Fraude de Tarjetas de Crédito")
st.write("Seleccione los valores de las variables para obtener la predicción de fraude.")

# Agregar sliders para las variables "Time" y "Amount"
slider_time = st.slider("Time:", float(X["Time"].min()), float(X["Time"].max()), float(X["Time"].mean()))
slider_amount = st.slider("Amount:", float(X["Amount"].min()), float(X["Amount"].max()), float(X["Amount"].mean()))

# Preparar los valores de entrada para la predicción
input_data = pd.DataFrame({"Time": [slider_time], "Amount": [slider_amount]})

# Realizar la predicción
prediction = model.predict(input_data)

# Mostrar el resultado de la predicción
if prediction ==1:
    st.error("¡Fraude detectado!")
else:
    st.success("Transacción segura")

# Guardar la predicción en un archivo pickle
with open('prediction.pkl', 'wb') as f:
    pickle.dump(prediction, f)

# Mostrar un mensaje de éxito
st.write('Archivo pickle creditcard')