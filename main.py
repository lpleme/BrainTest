import tensorflow as tf

# Loading the model
model = tf.keras.models.load_model('my_model.hdf5')

# Writing the header, description, and instructions
import streamlit as st

st.write("""
         # Brain Tumor Prediction TEST
         """
         )
st.write("This is a brain tumor classification web app to predict types of brain tumors")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

# Converting the uploaded picture to the size required by the model
import cv2
from PIL import Image, ImageOps
import numpy as np


def import_and_predict(image_data, model):
    size = (300, 300)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)) / 255.

    img_reshape = img_resize[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction


# This is where it'll spit out the prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    if np.argmax(prediction) == 0:
        st.write("glioma tumor")
    elif np.argmax(prediction) == 1:
        st.write("meningioma tumor")
    elif np.argmax(prediction) == 2:
        st.wrtie("no tumor")
    else:
        st.write("pituitary tumor")

    # Matrix with prediction results
    st.text("Probability (0: glimoa, 1: meningioma, 2: no tumor 3: pituitary")
    st.write(prediction)
