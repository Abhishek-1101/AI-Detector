import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model
model = tf.keras.models.load_model("model/model.h5")

IMG_SIZE = 128

st.title("🤖 AI vs Real Image Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.error(f"AI Generated ❌ ({prediction*100:.2f}%)")
    else:
        st.success(f"Real Image ✅ ({(1-prediction)*100:.2f}%)")