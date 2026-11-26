
import sys
import os
from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# --------------------------------------------------
# PROJECT PATH AND MODEL LOADING
# --------------------------------------------------

PROJECT_PATH = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_PATH / "forgery_classifier.h5"

model = tf.keras.models.load_model(str(MODEL_PATH))


# --------------------------------------------------
# GRAD-CAM FUNCTION
# --------------------------------------------------

def make_gradcam_heatmap(img_array, class_index):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer("top_conv").output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array, training=False
        )

        # Forged = 1, Real = 0
        if class_index == 1:
            loss = predictions[:, 0]
        else:
            loss = 1 - predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError("Unable to calculate Grad-CAM gradients.")

    pooled_grads = tf.reduce_mean(
        grads, axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()


# --------------------------------------------------
# HEATMAP OVERLAY
# --------------------------------------------------

def create_gradcam_overlay(image, heatmap):

    # Resize heatmap to original image dimensions
    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    )

    heatmap_image = heatmap_image.resize(
        image.size, Image.Resampling.BILINEAR
    )

    heatmap_array = np.asarray(
        heatmap_image, dtype=np.float32
    ) / 255.0

    original_array = np.asarray(
        image, dtype=np.float32
    )

    # Create a red heatmap
    colored_heatmap = np.zeros_like(original_array)

    colored_heatmap[:, :, 0] = heatmap_array * 255
    colored_heatmap[:, :, 1] = heatmap_array * 100

    # Blend original image and heatmap
    overlay = (
        0.6 * original_array +
        0.4 * colored_heatmap
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


# --------------------------------------------------
# STREAMLIT USER INTERFACE
# --------------------------------------------------

st.title("🔍 Image Forgery Detection")

st.write(
    "Upload an image to classify it as Real or Forged "
    "and visualize the model's Grad-CAM explanation."
)

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# IMAGE PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, caption="Original Image", width=350)

    # Preprocess image exactly as in the existing app
    resized_image = image.resize((256, 256))

    img_array = np.array(
        resized_image, dtype=np.float32
    )

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    pred = model.predict(img_array, verbose=0)[0][0]

    confidence = float(pred)

    if confidence < 0.5:

        class_index = 0

        st.success(
            f"🟢 REAL IMAGE "
            f"(Confidence: {(1-confidence)*100:.2f}%)"
        )

    else:

        class_index = 1

        st.error(
            f"🔴 FORGED IMAGE "
            f"(Confidence: {confidence*100:.2f}%)"
        )


    # --------------------------------------------------
    # GRAD-CAM VISUALIZATION
    # --------------------------------------------------

    st.subheader("Grad-CAM Visualization")

    try:

        heatmap = make_gradcam_heatmap(
            img_array, class_index
        )

        overlay_image = create_gradcam_overlay(
            image, heatmap
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )

        with col2:

            st.image(
                overlay_image,
                caption="Grad-CAM Heatmap Overlay",
                use_container_width=True
            )

        st.info(
            "The highlighted regions indicate areas that "
            "influenced the model's prediction. This is an "
            "explanation of model attention, not a definitive "
            "map of manipulated pixels."
        )

    except Exception as e:

        st.error(f"Grad-CAM error: {str(e)}")
