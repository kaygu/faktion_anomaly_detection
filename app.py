import streamlit as st
import tensorflow as tf
import numpy as np

from get_data import get_data
from model import build_model

import cv2


# @st.cache()
def load_model():
    try:
        model = tf.keras.models.load_model('model/myModel.h5')
    except:
        x, y = get_data()
        model = build_model(x, y)
        model.save("model/myModel.h5")
    return model

def main():
    st.title("Dice Anomaly Detection")

    # Step 1. Load models
    model = load_model()

    # Step 2. Draw the sidebar file uploader
    st.sidebar.title("Upload dice images to analyse them")
    uploaded_files = st.sidebar.file_uploader("Choose an image...", type="jpg", accept_multiple_files=True)
    

    # Step 4. Display model results / preformance in main page
    matrices = []
    if uploaded_files:
        for image in uploaded_files:
            # Transform StreamLit UploadedFile to np array matrix
            file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            matrices.append(opencv_image)
        matrices = np.asarray(matrices)
        matrices = matrices.reshape(-1,128,128, 1)
        results = model.predict(matrices)
        labels = []
        for result in results:
            labels.append("Defect" if int(result[0]) else "Normal")
        st.image(matrices, caption=labels, width=150)

if __name__ == '__main__':
    main()