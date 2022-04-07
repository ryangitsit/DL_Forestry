import tensorflow as tf
import keras
import numpy as np


def create_data():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "forest_data/Data/Train_Data/",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(256, 256),
    batch_size=32
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "forest_data/Data/Train_Data/",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=32
    )
    return train_ds,val_ds