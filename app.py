import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

st.header('Clasificación Fruta-Verdura')
model = load_model('./Modelo.keras')
data_cat = ['ajo',
 'anana',
 'arveja',
 'banana',
 'batata',
 'berenjena',
 'cebolla',
 'choclo',
 'coliflor',
 'espinaca',
 'granada',
 'jalapeno',
 'jengibre',
 'kiwi',
 'lechuga',
 'limon',
 'mango',
 'manzana',
 'melon',
 'morron',
 'nabo',
 'naranja',
 'papa',
 'paprika',
 'pepino',
 'pera',
 'rabano',
 'remolacha',
 'repollo',
 'sandia',
 'soja',
 'tomate',
 'uva',
 'zanahoria']

frutas = ['anana', 'banana', 'granada', 'kiwi', 'limon', 'mango', 'manzana', 'melon', 'naranja', 'paprika', 'pera', 'sandia', 'tomate', 'uva']
verduras = ['ajo', 'arveja', 'batata', 'berenjena', 'cebolla', 'choclo', 'coliflor', 'espinaca', 'jengibre', 'lechuga', 'nabo', 'rabano', 'remolacha', 'repollo', 'soja', 'zanahoria', 'papa']

img_height = 180
img_width = 180

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, width=200)

    image = image.resize((img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image)
    img_bat = tf.expand_dims(img_arr, 0)

    predict = model.predict(img_bat)

    score = tf.nn.softmax(predict[0])
    predicted_class_idx = np.argmax(score)
    predicted_class = data_cat[predicted_class_idx]

    if predicted_class in frutas:
        tipo = 'fruta'
    elif predicted_class in verduras:
        tipo = 'verdura'
    else:
        tipo = 'desconocido'

    st.write('La imagen parece ser: ' + tipo)
    st.write('La verdura o fruta de la imagen es: ' + predicted_class)
    st.write('Con una efectividad de {:.2f}%'.format(np.max(score) * 100))
