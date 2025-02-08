import streamlit as st
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Ensure Git LFS files are pulled
os.system("git lfs pull")

# Streamlit UI Title
st.title("Waste Classification Model")
st.write("Upload an image to classify it as **Recyclable** or **Organic Waste**")

# Define Image Size
IMG_SIZE = (224, 224)

# Load Pretrained Model (Ensure 'waste_classifier.h5' exists in the same directory)
def load_trained_model():
    model_path = "waste_classifier.h5"
    if not os.path.exists(model_path):
        st.error(f"Error: Model file '{model_path}' not found. Make sure it's correctly uploaded and pulled from Git LFS.")
        return None
    return load_model(model_path)

model = load_trained_model()

# File Uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

# Function to classify the uploaded image
def classify_image(img):
    img = img.resize(IMG_SIZE)  # Resize image to match model input
    img_array = np.array(img)  # Convert to NumPy array
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    result = np.argmax(prediction)
    labels = ["Recyclable Waste", "Organic Waste"]
    return labels[result]

if uploaded_file is not None:
    # Convert the uploaded image to PIL format
    image = Image.open(uploaded_file).convert("RGB")

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform classification if model is loaded
    if model is not None:
        result = classify_image(image)
        st.write(f"### Predicted Category: {result}")
    else:
