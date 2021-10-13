import io
import threading
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import keras
import numpy as np
from PIL import Image
import tensorflow as tf
from tika import parser
import cv2


@st.cache(allow_output_mutation=True)
def loadExamples():
    classes = joblib.load('./data/train_classes.pkl')
    examples = joblib.load('./data/train_examples.pkl')
    return classes, examples


@st.cache(allow_output_mutation=True)
def thresh_values():
    return joblib.load('./data/values_thresh_mask.pkl')


@st.cache(allow_output_mutation=True)
def loadModel():
    return keras.models.load_model('models/cnn_11_classificator.h5')


model = loadModel()
train_classes, train_img_ex = loadExamples()
df = thresh_values()
st.sidebar.title("DICE ANOMALY DETECTION")
st.sidebar.write('\n')
uploaded_file = st.sidebar.file_uploader('Please Upload the image of the dice', type="jpg")

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('ANOMALY DETECTION')


def plot_from_class(prediction):
    fig, ax = plt.subplots()
    predicted_label = np.argmax(prediction)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    ax.imshow(np.resize(train_img_ex[train_img_index], (128, 128, 1)), cmap='gray', vmin=0, vmax=255)
    class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.xlabel("Classified in class : {}".format(class_names[predicted_label]))
    return fig


def apply_mask(img_arr, prediction, img_size=128):
    resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
    _, thresh = cv2.threshold(resized_arr, 50, 255, cv2.THRESH_BINARY_INV)
    min = df[['min', 'max']].iloc[np.argmax(prediction)][0]
    max = df[['min', 'max']].iloc[np.argmax(prediction)][1]
    meanpix = 'Mean pixel: {:.2f}'.format(np.mean(thresh))

    return min < np.mean(thresh) < max, meanpix, thresh


if uploaded_file is not None:
    data = uploaded_file.read()
    dataBytesIO = io.BytesIO(data)
    img = Image.open(dataBytesIO)
    img = img.convert('L')
    img = np.array(img)

    prediction = model.predict(img[None, :, :])
    train_img_index = np.where(train_classes == np.argmax(prediction))[0]
    col1, col2, col3= st.beta_columns(3)
    assert_mask, meanpix, thresh = apply_mask(img[None, :, :], prediction)

    with col1:
        st.write('Uploaded dice:')
        fig, ax = plt.subplots()
        predicted_label = np.argmax(prediction)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        st.pyplot(fig)
        st.markdown(meanpix)


    with col2:
        st.write('Predicted class')
        st.pyplot(plot_from_class(prediction))
        st.markdown('Class mean pixel: {:.2f}'.format(df[['mean']].iloc[np.argmax(prediction)][0]))

    if assert_mask:
        st.info('NORMAL DICE')
    else:
        st.warning('ANOMALOUS DICE')
