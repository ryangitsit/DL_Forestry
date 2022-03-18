import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np


import logger
import os

from pathlib import Path

log = logger.setup_logger(__name__)

def create_dataset(config, val_split = 0.2):
    log.info("Loading dataset...")

    num_skipped = 0
    for folder_name in ("Fire", "Non_Fire"):
        #folder_path = os.path.join("C:","Users","rmolo","Desktop","DLForestry","DL_Forestry", "forest_data", "Train_Data", folder_name)
        #folder_path = "Users\rmolo\Desktop\DLForestry\DL_Forestry\forest_data\Train_Data"
        folder_path = Path("forest_data","Data","Train_Data")
        # for fname in os.listdir(folder_path):
        #     fpath = os.path.join(folder_path, fname)
        #     try:
        #         fobj = open(fpath, "rb")
        #         is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        #     finally:
        #         fobj.close()

        #     if not is_jfif:
        #         num_skipped += 1
        #         # Delete corrupted image
        #         os.remove(fpath)

    print("Deleted %d images" % num_skipped)
    image_size = (180, 180)
    batch_size = 32

    if config.crossvalidation:
        log.warning("Crossvalidation is used, but not implemented in this way, turn around now!")    
    kf = KFold(n_splits = 5)

    filenames = []
    labels = []
      

    for file in os.listdir('forest_data/Data/Train_Data/Fire'):
        filenames.append(os.path.join('Fire', file))
        labels.append('Fire')

    for file in os.listdir('forest_data/Data/Train_Data/Non_Fire'):
        filenames.append(os.path.join('Non_Fire', file))
        labels.append('Non_Fire')

    print ("#################",len(filenames), len(labels), "#################")

    d = {'filename': filenames, 'label': labels}
    alldata = pd.DataFrame(d)
    alldata = alldata.sample(frac=1).reset_index(drop=True) 
    Y = alldata[['label']]

    #optional augmentation applied through idg:
    if config.augmentation:
        idg = ImageDataGenerator(width_shift_range=0.0,
                            height_shift_range=0.0,
                            zoom_range=0.0,
                            rotation_range=36,
                            fill_mode='nearest',
                            horizontal_flip = True,
                            rescale=None)
    else:
        idg = ImageDataGenerator(width_shift_range=0.0,
                            height_shift_range=0.0,
                            zoom_range=0.0,
                            fill_mode='nearest',
                            horizontal_flip = True,
                            rescale=None)

    val_idg = keras.preprocessing.image.ImageDataGenerator()
  

    for train_index, val_index in kf.split(np.zeros(len(Y)), Y):
        training_data = alldata.iloc[train_index]
        validation_data = alldata.iloc[val_index]
        print ("#################data",len(training_data), len(validation_data), "#################")
        train_ds = idg.flow_from_dataframe(training_data, target_size = (180, 180), directory = 'forest_data/Train_Data', x_col = "filename", y_col = "label", class_mode = "categorical", shuffle = True)
        val_ds  = val_idg.flow_from_dataframe(validation_data, target_size = (180, 180), directory = 'forest_data/Train_Data', x_col = "filename", y_col = "label", class_mode = "categorical", shuffle =True)
        print ("#################data",len(train_ds), len(val_ds), "#################")
        break


    return (train_ds, val_ds)