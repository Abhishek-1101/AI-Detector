import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("model/model.h5")

IMG_SIZE = 128

def predict_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        print("❌ Error: Image not found or path incorrect")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return "AI Generated ❌", prediction
    else:
        return "Real Image ✅", 1 - prediction

# Test image
result, confidence = predict_image("test.jpg")

print("Result:", result)
print("Confidence:", confidence)