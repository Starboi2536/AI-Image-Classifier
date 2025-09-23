import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

#Load the saved model
model = tf.keras.models.load_model("model/image_classifier_model.h5")

#Load and preprocess custom
img_path ="custom_image/my.test.jpg" # Change this to your image path
img = image.load_img(img_path, target_size=(180, 180))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) #Add batch dimension

#Predict
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
class_name = ['cats', 'dogs'] #same order as your dataset folders
print(f"This image most likely belongs to: {class_name[np.argmax(score)]} with a {100 * np.max(score):.2f}% confidence.")