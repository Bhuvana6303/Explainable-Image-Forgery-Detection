import tensorflow as tf
from preprocess import load_image

# Load your actual model file
model = tf.keras.models.load_model(
    "/Users/venkatsomanadh/Downloads/Forgery Detection/forgery_classifier.h5"
)

def predict_image(img_path):
    img = load_image(img_path)
    img = tf.expand_dims(img, 0)

    pred = model.predict(img)[0][0]

    return "FORGED IMAGE" if pred >= 0.5 else "REAL IMAGE"


