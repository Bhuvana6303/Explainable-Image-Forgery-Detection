import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from src.preprocess import load_image

MODEL_PATH = "/Users/venkatsomanadh/Downloads/Forgery Detection/forgery_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)

st.title("🔍 Image Forgery Detection")
st.write("Upload an image and the model will classify it as **Real** or **Forged**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    img_array = np.array(image.resize((256, 256))) / 255.0
    img_array = np.expand_dims(img_array, 0)

    pred = model.predict(img_array)[0][0]
    confidence = float(pred)

    if pred < 0.5:
        st.success(f"🟢 REAL IMAGE (Confidence: {100 - confidence*100:.2f}%)")
    else:
        st.error(f"🔴 FORGED IMAGE (Confidence: {confidence*100:.2f}%)")
