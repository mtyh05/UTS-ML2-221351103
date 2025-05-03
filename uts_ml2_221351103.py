import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="astronomical-data.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Klasifikasi Spektral Bintang")
st.write("Masukkan parameter bintang berdasarkan data astronomi.")

# Input numerik
temperature = st.number_input("Temperature (K)", min_value=1000.0, max_value=50000.0, value=5000.0)
luminosity = st.number_input("Luminosity (L/Lo)", min_value=0.0001, max_value=100000.0, value=1.0)
radius = st.number_input("Radius (R/Ro)", min_value=0.01, max_value=1000.0, value=1.0)
absolute_magnitude = st.number_input("Absolute Magnitude (Mv)", min_value=-10.0, max_value=20.0, value=4.83)

# Input kategori (encoded)
star_type = st.selectbox("Star Type", ['Brown Dwarf', 'Red Dwarf', 'White Dwarf', 'Main Sequence', 'Supergiant', 'Hypergiant'])
star_color = st.selectbox("Star Color", ['Blue', 'Blue White', 'White', 'Yellowish White', 'Pale Yellow Orange', 'Orange', 'Orange Red'])

type_dict = {'Brown Dwarf': 0, 'Red Dwarf': 1, 'White Dwarf': 2, 'Main Sequence': 3, 'Supergiant': 4, 'Hypergiant': 5}
color_dict = {'Blue': 0, 'Blue White': 1, 'White': 2, 'Yellowish White': 3, 'Pale Yellow Orange': 4, 'Orange': 5, 'Orange Red': 6}

encoded_type = type_dict[star_type]
encoded_color = color_dict[star_color]

# Tambahkan fitur dummy sebagai pengganti Spectral Class (misalnya 0)
dummy_spectral_class = 0

# Gabungkan semua fitur menjadi input (7 fitur)
input_data = np.array([[temperature, luminosity, radius, absolute_magnitude, encoded_type, encoded_color, dummy_spectral_class]])

if st.button("Prediksi Spektral Class"):
    try:
        input_scaled = scaler.transform(input_data).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_scaled)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        predicted_label = np.argmax(prediction)
        spectral_class = label_encoder.inverse_transform([predicted_label])[0]
        
        st.success(f"Kelas Spektral Bintang: **{spectral_class.upper()}**")
    except ValueError as e:
        st.error(f"Error saat transformasi data: {e}")
