import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load scaler dan model TFLite
scaler = joblib.load("scaler.pkl")
interpreter = tf.lite.Interpreter(model_path="iris_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Kelas iris
class_names = ['Setosa', 'Versicolor', 'Virginica']

# UI Streamlit
st.set_page_config(page_title="🌸 Prediksi Bunga Iris", layout="centered")
st.title("🌸 Prediksi Spesies Bunga Iris (TFLite)")
st.markdown("Masukkan fitur bunga iris untuk memprediksi spesiesnya secara otomatis!")

# Input pengguna
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
with col2:
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

# Prediksi
if st.button("🔍 Prediksi"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled_data = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], scaled_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_index = np.argmax(prediction)
    pred_class = class_names[pred_index]
    confidence = prediction[pred_index] * 100

    st.success(f"🌼 Model memprediksi: **{pred_class}** dengan keyakinan {confidence:.2f}%")
    st.balloons()
