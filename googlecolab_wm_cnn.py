import streamlit as st
import numpy as np
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

# Check if the model file exists
model_path = "waste_classifier.h5"
if not os.path.exists(model_path):
    st.error(f"❌ Model file '{model_path}' is missing. Make sure it's correctly uploaded and pulled from Git LFS.")
    os.system("ls -lh")  # Debug: List all files in the current directory
    os.system("git lfs ls-files")  # Debug: Check if the model is in Git LFS
    model = None
else:
    st.success(f"✅ Model file '{model_path}' found!")

    # Load Pretrained Model
    @st.cache_resource
    def load_trained_model():
        return load_model(model_path)  # Directly load the saved model

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
        st.error("❌ Model loading failed. Check if 'waste_classifier.h5' is available.")
