import tensorflow as tf
import keras
import numpy as np


def create_model(pooling = False, act = "softmax"):
    model = tf.keras.applications.InceptionV3(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(256,256,3),
        pooling=pooling,
        classes=2,
        classifier_activation=act)
    return model


def model_stuff(train_ds, val_ds, config):
    ep=config.epochs
    lr=config.learningrate
    pooling = False
    act = "softmax"
    opt=config.optimizer
    mo=config.momentum
    # if necessary or wanted create model
    model = create_model(pooling, act)
    if opt=="adam":
        # compile models TODO: play with loss function, find out why sparse is the only one withour an error
        modelAdam = tf.keras.models.clone_model(model)
        modelAdam.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr,momentum=mo),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
        print("modelAdam")
        modelAdam.fit(train_ds, epochs= ep, callbacks=None, validation_data=val_ds)

    if opt=="SGD":
        modelSGD = tf.keras.models.clone_model(model)
        modelSGD.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=lr,momentum=mo), #idea:learning rate schedule (tf.keras.optimizers.schedules.LearningRateSchedule) so it starts with 0.01 and gets smaller over time
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
        print("modelSGD")
        modelSGD.fit(train_ds, epochs= ep, callbacks=None, validation_data=val_ds)

    if opt=="RMS":
        modelRMSprop = tf.keras.models.clone_model(model)
        modelRMSprop.compile(
            tf.keras.optimizers.RMSprop(learning_rate=lr,momentum=mo), # momentum could be played with \o/
            loss="sparse_categorial_crossentropy",
            metrics=["accuracy"])
        print("modelRMSprop")
        modelRMSprop.fit(train_ds, epochs= ep, callbacks=None, validation_data=val_ds)

    # # fit the different models :D
    # print("modelAdam")
    # modelAdam.fit(train_ds, epochs= ep, callbacks=None, validation_data=val_ds)
    # print("modelSGD")
    # modelSGD.fit(train_ds, epochs= ep, callbacks=None, validation_data=val_ds)
    # print("modelRMSprop")
    # modelRMSprop.fit(train_ds, epochs= ep, callbacks=None, validation_data=val_ds)

    return model #could return the trained models -> returning unchanged one now!