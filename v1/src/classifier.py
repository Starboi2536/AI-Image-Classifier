import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

#Set path to training and validation data
train_dir = "dataset/train"
val_dir = "dataset/val"

#Load training dataset from directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(180, 180),
    batch_size=32  

)
 
#Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(180, 180),
    batch_size=32
)

#optional to view code 

class_name = train_ds.class_names #Gets the class labels eg, Cat/dog

#displays the first batch of 9 images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8")) #shows the image
        plt.title(class_name[labels[i]]) # Title is class label
        plt.axis("off") #Hide exits
        plt.show()

#Build and train a CNN model
model = Sequential([
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    layers.Conv2D(32,3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_name )) # Output layer with number of classes
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_ds, validation_data=val_ds, epochs=5)

#Save the trained model
model.save ("model/image_classifier_model.h5")