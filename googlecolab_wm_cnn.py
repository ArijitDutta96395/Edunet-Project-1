import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Install Git LFS manually (for non-root users)
os.system("curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash")
os.system("sudo apt-get install git-lfs")
os.system("git lfs install")
os.system("git lfs pull")

# Debugging: Verify the model file size
os.system("ls -lh waste_classifier.h5")

# Debugging: List files to check if the model is present
st.write("📂 Listing files in the repository:")
os.system("ls -lh")

# Define Model Path
model_path = "waste_classifier.h5"

# Check if the model file exists
if not os.path.exists(model_path):
    st.error(f"❌ Model file '{model_path}' is missing. Ensure it's uploaded and pulled from Git LFS.")
    os.system("git lfs ls-files")  # Debug: Check if it's in Git LFS
    model = None
else:
    st.success(f"✅ Model file '{model_path}' found!")

    # Load Model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        model = None

# UI
st.title("Waste Classification Model")
st.write("Upload an image to classify it as **Recyclable** or **Organic Waste**")

# File Uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

# Image Classification Function
def classify_image(img):
    IMG_SIZE = (224, 224)
    img = img.resize(IMG_SIZE)  # Resize
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    result = np.argmax(prediction)
    labels = ["Recyclable Waste", "Organic Waste"]
    return labels[result]

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform classification only if the model is loaded
    if model is not None:
        result = classify_image(image)
        st.write(f"### Predicted Category: {result}")
    else:
        st.error("❌ Model loading failed. Ensure 'waste_classifier.h5' is correctly uploaded.")
