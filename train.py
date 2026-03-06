import tensorflow as tf
import streamlit as st
from model.cnn_model import create_model


class StreamlitProgress(tf.keras.callbacks.Callback):

    def __init__(self, epochs):
        self.epochs = epochs
        self.progress_bar = st.progress(0)

    def on_epoch_end(self, epoch, logs=None):
        progress = int((epoch + 1) / self.epochs * 100)
        self.progress_bar.progress(progress)


def train_model(train_data, val_data, num_classes, epochs=50):

    model = create_model(num_classes)

    progress_callback = StreamlitProgress(epochs)

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[progress_callback]
    )

    return model