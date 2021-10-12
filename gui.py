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
    st.title("Dice Factory Anomaly Detection")
    uploaded_files = st.sidebar.file_uploader("Choose an image...", type="jpg", accept_multiple_files=True)

    # Step 1. Load data ?
    

    # Step 2. Load models
    model = load_model()

    # Step 3. Draw the sidebar UI.
    
    features = ...  # Internally, this uses st.sidebar.slider(), etc.

    # Step 4. Draw main page
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
    # print(matrices)
    # for i, image in enumerate(matrices):
    #     label = "Defect" if int(results[i][0]) else "Normal"
    
        # st.text(results[i])
        # st.image(images[0], caption='Uploaded Image.', use_column_width=True)
        # st.write("")
        # st.write("Classifying...")
        # label = predict(uploaded_file)
        # st.write('%s (%.2f%%)' % (label[1], label[2]*100))

    # Step 5. Display model results / preformance

if __name__ == '__main__':
    main()