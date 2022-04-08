from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pandas as pd


def load_aug_data():
    aug = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=20,
        zoom_range=0.15,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2
    )

    train_ds = aug.flow_from_directory('forest_data/Data/Train_Data/', target_size=(256, 256), batch_size=32, subset='training', class_mode='binary')
    val_ds = aug.flow_from_directory('forest_data/Data/Train_Data/', target_size=(256, 256), batch_size=32, subset='validation', class_mode='binary')

    return train_ds, val_ds