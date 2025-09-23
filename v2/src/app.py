import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os


# Load model
MODEL_PATH = "model/v2_plant_classifier_model.keras"
IMG_SIZE = (128, 128)

#streamlit UI
st.set_page_config(page_title="Plant Classifier V2", page_icon="🌱", layout="centered")
st.markdown("""
This app classifies images of **Flowers** into five categories:
- 🌼 Daisy  
- 🌻 Dandelion  
- 🌹 Rose  
- 🌞 Sunflower  
- 🌷 Tulip

Upload an image below and let the model do the magic 🪄
""")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

#Class labels
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.success(f" Prediction: **{predicted_class}** ({confidence:.2f}% confidence)")