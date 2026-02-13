# AI Image Classifier (Cats vs Dogs)

This is a beginner **AI Image Classifer** built with **TensorFlow** and **Streamlit**.
The project demonstrates how to train and deploy a simple deep learning model that can distinguish between **cats** and **dogs**.

---

## Features
-Upload an image (`jpg`, `jpeg`, `png`)
-Classifies the image as either **Cat** or **Dog**
-Shows prediction confidence %
simple Streamlit web interface

---

## Tech Stack
- **Python 3.10+**
- **TensorFlow 2.x**
- **Streamlit**
- **NumPy**
- **Pillow**


---

## How to run Loacally
**1. Clone the repo**

git clone https://github.com/Starboi2536/AI-Image-Classifier.git
cd AI-Image-Classifier


**2. Create a virtual environment(recommended)**
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

**3. Install dependencies **
pip install -r requirements.txt

**4. Download/Train the model**
 please not that the trained model (image_classifer_model.h5) is not included in this repo as the file was to large for Github
Therefore the options are:
  -Download from link ()
  -Train your own model and save it inside the model/ folder: "model.save("model/image_classifier_model.h5")" 

**Run the app**
streamlit run app.py
---

Project Structure

AI-Image-Classifier/
-app.py                  # Main Streamlit app
-model/                  # Folder where trained model should be placed
-requirements.txt        # Project dependencies
-README.md               # Project documentation

---

NOTE!!!
-The trained model file (.h5) is too large for GitHub(>100).
-A download link or training script will be provided later (in progress)

---

Future Improvements
-Extend to multi-classification (classify plants)
-Improve accuracy with data augmentation

---

Auther
-Jonathan Malunga
-LinkedIn:www.linkedin.com/in/jonathan-malunga-27b411230
-GitHub:https://github.com/Starboi2536
