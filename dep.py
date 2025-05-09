# app.py

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title("Breast Cancer classification")
model = load_model("breast_cancer_model.h5")

st.title("Breast Cancer Detection")
st.write("Upload an image and the model will predict if it's benign or malignant.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)[0][0]
    st.write(f"Prediction Score: {prediction:.2f}")

    if prediction > 0.5:
        st.error("Malignant (Hard Cancer detected)")
    else:
        st.success("Benign (Soft cancer detected)")
