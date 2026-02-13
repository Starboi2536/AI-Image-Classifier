import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "v2_plant_classifier_model.keras")

IMG_SIZE = (128, 128)



# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(
    page_title="Plant Classifier V2",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Plant Classifier V2")

st.markdown("""
This app classifies images of **Flowers** into five categories:

- 🌼 Daisy  
- 🌻 Dandelion  
- 🌹 Rose  
- 🌞 Sunflower  
- 🌷 Tulip  

Upload an image below and let the model do the magic 🪄
""")


# ---------------------------
# Load Model (Cached)
# ---------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please run train.py first.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ---------------------------
# Class Labels
# ---------------------------
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]


# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload a plant image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("Analyzing image..."):
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.success(
        f"Prediction: **{predicted_class}** ({confidence:.2f}% confidence)"
    )
