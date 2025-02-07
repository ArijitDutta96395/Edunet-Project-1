import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import tempfile
import os

# Streamlit UI Title
st.title("Waste Classification Model")
st.write("Upload an image to classify it as **Recyclable** or **Organic Waste**")

# Define Image Size
IMG_SIZE = (224, 224)

# Load Pretrained Model (Ensure you have a trained model as 'waste_classifier.h5')
@st.cache_resource
def load_trained_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.load_weights("waste_classifier.h5")  # Ensure the trained model weights exist
    return model

model = load_trained_model()

# File Uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

def classify_image(img):
    img = cv2.resize(img, IMG_SIZE)
    img = np.reshape(img, (-1, 224, 224, 3)) / 255.0  # Normalize
    prediction = model.predict(img)
    result = np.argmax(prediction)
    labels = ["Recyclable Waste", "Organic Waste"]
    return labels[result]

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Display Image
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Perform Classification
    result = classify_image(img)
    st.write(f"### Predicted Category: {result}")
