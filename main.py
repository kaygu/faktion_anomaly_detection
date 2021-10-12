import tensorflow as tf

from get_data import get_data
from model import build_model

if __name__ == "__main__":
    x, y = get_data()
    try:
        model = tf.keras.models.load_model('model/myModel.h5')
    except:
        model = build_model(x, y)
        model.save("model/myModel.h5")
    y_pred = model.predict(x)
    cm = tf.math.confusion_matrix(y, y_pred)
    print(cm)