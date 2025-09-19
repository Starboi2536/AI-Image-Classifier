import tensorflow_datasets as tfds
import tensorflow as tf
import os
import shutil


#Dataset paths
dataset_dir = "dataset_v2"
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")

#Remove old dataset if exist
if os.path.exists(dataset_dir):
    shutil.rmtree(dataset_dir)

#Load dataset (80% train, 20% validation)
(ds_train, ds_val), ds_info= tfds.load(
    "tf_flowers",
    split=["train[:80%]", "train[:80%]"],
    as_supervised=True,
    with_info=True
)

class_names = ds_info.features["label"].names
print("Classes:", class_names)

#Save dataset as images
def save_dataset(dataset, base_dir):
    for i, (image, label) in enumerate(tfds.as_numpy(dataset)):
        class_name = class_names[label]
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(class_dir, f"{class_name}_{i}.jpg")
        tf.keras.utils.save_img(img_path, image)

print("Saving training set...")
save_dataset(ds_train, train_dir)

print("Saving validation set...")
save_dataset(ds_val, val_dir)

print("Dataset ready at:", os.path.abspath(dataset_dir))