import tensorflow as tf
from tensorflow import keras
import logger
from inception_custom import InceptionCustom
import pandas as pd
import time

log = logger.setup_logger(__name__)

image_size = (180, 180)
batch_size = 32


def create_model(config, train_ds, val_ds, kfold = 0):
   

    if config.activation == 'elu':
        model = InceptionCustom(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=image_size + (3,),
            pooling=None,
            classes=2,
            classifier_activation='softmax',
            act=config.activation
        )
    else:
        model = tf.keras.applications.InceptionV3(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=image_size + (3,),
            pooling=None,
            classes=2,
            classifier_activation="softmax"
        )

    #make_model(input_shape=image_size + (3,), num_classes=2)
    #keras.utils.plot_model(model, show_shapes=True)

  
    if config.crossvalidation:
        tmp = f"model_checkpoints/Foldnr{kfold}_{config.epochs}Epochs_{config.activation}Activation_{config.optimizer}Optimizer-{config.augmentation}Augmentation_lrate{config.learningrate}_mom{config.momentum}_"
        callbacks = [
            keras.callbacks.ModelCheckpoint(tmp + "save_at_{epoch}.h5", period = 50),
        ]
    else:
        tmp = f"model_checkpoints/{config.epochs}Epochs_{config.activation}Activation_{config.optimizer}Optimizer-{config.augmentation}Augmentation_lrate{config.learningrate}_mom{config.momentum}_"
        callbacks = [
            keras.callbacks.ModelCheckpoint(tmp + "save_at_{epoch}.h5", period = 50),
        ]


    if config.optimizer == 'sgdm':
        model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=config.learningrate, momentum=config.momentum, nesterov=False, name='SGD'
                ),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
    else:
        model.compile(
            tf.keras.optimizers.RMSprop(
            learning_rate=config.learningrate, rho=0.9, momentum=config.momentum, epsilon=1.0, centered=False,
                name='RMSprop'
                ),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    print ("#################",len(train_ds), len(val_ds), "#################")

    # train the model and record the history of relevant data over the epochs
    begintime = time.time()
    hist = model.fit(
        train_ds, epochs=config.epochs, callbacks=callbacks, validation_data=val_ds,
    )
    endtime = time.time()
    lapsedtimemillis = round((endtime - begintime) * 1000)
    log.info(f"Training completed in {lapsedtimemillis} milliseconds")

    log.info("Saving history of accuracy...")                                       
    hist_df = pd.DataFrame.from_dict(hist.history) 
    if config.crossvalidation:
        filepath = f"./model_training_history/Foldnr{kfold}_{config.epochs}Epochs_{config.activation}Activation-{config.optimizer}Optimizer-{config.augmentation}Augmentation_lrate{config.learningrate}_mom{config.momentum}" 
    else:
        filepath = f"./model_training_history/{config.epochs}Epochs_{config.activation}Activation-{config.optimizer}Optimizer-{config.augmentation}Augmentation_lrate{config.learningrate}_mom{config.momentum}"
    hist_df.to_csv(filepath + "-history.csv")

    # write lapsed time in millis to file:
    f = open(filepath + "-lapsedtimemillis.txt", "a")
    f.write(f"Training took: \n{lapsedtimemillis}\nmilliseconds")
    f.close()

    return model
    
    
