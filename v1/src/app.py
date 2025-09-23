import streamlit as st 
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL.Image

#Load Model
model = tf.keras.models.load_model("model/image_classifier_model.h5")
class_names = ['cats', 'dogs']

#Title
st.title ("Beginner AI Image Classifier - Cat or Dog")

#Upload image
uploaded_file =st.file_uploader("Upload an image...", type=["jpg","jpeg", "png"])

if uploaded_file is not None:
    img =PIL.Image.open(uploaded_file).resize ((180,180))
    st.image(img, caption= "Uploaded Image", use_column_width=True)

    #Preprocess
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0 
    img_array = tf.expand_dims(img_array, 0) #Batch dimension

    #Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_label = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    #Show result
    st.success(f"Prediction: **{predicted_label}** ({confidence:.2f}% confidence)")