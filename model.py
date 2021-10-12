
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model(x, y):

    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.15, random_state=123)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=123)

    # Model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(128, 128, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compiling model
    model.compile(
      # Choose the loss function
      loss='binary_crossentropy',
      # Choose your optimizer
      optimizer='adam',
      # Choose the metric the model will use to evaluate his learning
      metrics=['accuracy']
    )

    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Flow training images in batches of 64 using train_datagen generator
    train_generator = train_datagen.flow(
        # This is the source directory for training images    
        x_train,
        y_train,
        # Define how big are gonna be your batch.
        batch_size=64,
        seed=0
    )
    validation_generator = validation_datagen.flow(
        # This is the source directory for validation images    
        x_val,
        y_val,
        # Define how big are gonna be your batch.
        batch_size=64,
        seed=0
    )
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('accuracy') > .99:
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=50,
        callbacks=[myCallback()],
    )

    print(history.epoch, history.history['accuracy'][-1])
    model.evaluate(x_test, y_test)
    return model