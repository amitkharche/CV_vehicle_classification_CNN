"""
Streamlit app to classify uploaded vehicle images.
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image

st.set_page_config(page_title="🚗 Vehicle Type Classifier", layout="centered")
st.title("🚗 Vehicle Type Classifier from CCTV Image")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/vehicle_cnn_model.h5")
    with open("model/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

def preprocess_image(image):
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

model, label_encoder = load_model()

uploaded_file = st.file_uploader("Upload a vehicle image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=False, width=250)
    input_tensor = preprocess_image(image)
    prediction = model.predict(input_tensor)[0]
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    st.markdown("### 🧠 Prediction:")
    st.markdown(f"## `{predicted_label.upper()}`")
    st.bar_chart(prediction)
