import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import time


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
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
        print("modelAdam")
        tmp = f"model_checkpoints/{config.epochs}Epochs__{config.optimizer}Optimizer-{config.augmentation}Augmentation_lrate{config.learningrate}_mom{config.momentum}_"
        callbacks = [
            keras.callbacks.ModelCheckpoint(tmp + "save_at_{epoch}.h5", period = 50),
        ]
        begintime = time.time()
        hist = modelAdam.fit(train_ds, epochs= ep, callbacks=callbacks, validation_data=val_ds)

    elif opt=="SGD":
        modelSGD = tf.keras.models.clone_model(model)
        modelSGD.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=lr,momentum=mo), #idea:learning rate schedule (tf.keras.optimizers.schedules.LearningRateSchedule) so it starts with 0.01 and gets smaller over time
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
        print("modelSGD")
        tmp = f"model_checkpoints/{config.epochs}Epochs__{config.optimizer}Optimizer-{config.augmentation}Augmentation_lrate{config.learningrate}_mom{config.momentum}_"
        callbacks = [
            keras.callbacks.ModelCheckpoint(tmp + "save_at_{epoch}.h5", period = 50),
        ]
        begintime = time.time()
        hist = modelSGD.fit(train_ds, epochs= ep, callbacks=callbacks, validation_data=val_ds)

    elif opt=="RMS":
        modelRMSprop = tf.keras.models.clone_model(model)
        modelRMSprop.compile(
            tf.keras.optimizers.RMSprop(learning_rate=lr,momentum=mo), # momentum could be played with \o/
            loss="sparse_categorial_crossentropy",
            metrics=["accuracy"])
        print("modelRMSprop")
        tmp = f"model_checkpoints/{config.epochs}Epochs__{config.optimizer}Optimizer-{config.augmentation}Augmentation_lrate{config.learningrate}_mom{config.momentum}_"
        callbacks = [
            keras.callbacks.ModelCheckpoint(tmp + "save_at_{epoch}.h5", period = 50),
        ]
        begintime = time.time()
        hist = modelRMSprop.fit(train_ds, epochs= ep, callbacks=callbacks, validation_data=val_ds)
    else:
        print("error: choose optimizer")

    # recording
    # begintime = time.time()
    # hist = model.fit(
    #     train_ds, epochs=config.epochs, callbacks=callbacks, validation_data=val_ds,
    # )
    endtime = time.time()
    lapsedtimemillis = round((endtime - begintime) * 1000)
    print(f"Training completed in {lapsedtimemillis} milliseconds")

    print("Saving history of accuracy...")                                       
    hist_df = pd.DataFrame.from_dict(hist.history) 
    filepath = f"./model_training_history/{config.epochs}Epochs_-{config.optimizer}Optimizer-{config.augmentation}Augmentation_lrate{config.learningrate}_mom{config.momentum}"
    hist_df.to_csv(filepath + "-history.csv")

    # write lapsed time in millis to file:
    f = open(filepath + "-lapsedtimemillis.txt", "a")
    f.write(f"Training took: \n{lapsedtimemillis}\nmilliseconds")
    f.close()

    return model #could return the trained models -> returning unchanged one now!