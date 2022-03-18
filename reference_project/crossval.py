import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, StratifiedKFold
import inception_model as inception
import dataset as ds
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logger
import matplotlib.pyplot as plt

log = logger.setup_logger(__name__)


def evaluate(config):
    log.info("Beginning crossvalidation...")

    kf = KFold(n_splits = 5)

    filenames = []
    labels = []
      

    for file in os.listdir('forest_data/Train_Data/Fire'):
        filenames.append(os.path.join('Fire', file))
        labels.append('Fire')

    for file in os.listdir('forest_data/Train_Data/Non_Fire'):
        filenames.append(os.path.join('Non_Fire', file))
        labels.append('Non_Fire')



    d = {'filename': filenames, 'label': labels}
    alldata = pd.DataFrame(d)
    alldata = alldata.sample(frac=1).reset_index(drop=True) 
    Y = alldata[['label']]

    #Any augmentation could be performed here
    if config.augmentation:
        log.warning("The augmentation is desired, but not yet implemented for crossvalidation")
    #TODO implement augmentation
    idg = ImageDataGenerator(width_shift_range=0.0,
                         height_shift_range=0.0,
                         zoom_range=0.0,
                         fill_mode='nearest',
                         horizontal_flip = False,
                         rescale=None)

    foldnr = 0
    
    for train_index, val_index in kf.split(np.zeros(len(Y)), Y):
        training_data = alldata.iloc[train_index]
        validation_data = alldata.iloc[val_index]

        train_data_generator = idg.flow_from_dataframe(training_data, target_size = (180, 180), directory = 'PetImages', x_col = "filename", y_col = "label", class_mode = "categorical", shuffle = False)
        val_data_generator  = idg.flow_from_dataframe(validation_data, target_size = (180, 180), directory = 'PetImages', x_col = "filename", y_col = "label", class_mode = "categorical", shuffle =False)

        foldnr = foldnr + 1
        model = inception.create_model(config, train_data_generator, val_data_generator , kfold = foldnr)

        if foldnr == 1:
            results = np.array(model.evaluate(val_data_generator))
        else:
            results += np.array(model.evaluate(val_data_generator))
        
        
    results = results / 5.0
    #results = dict(zip(model.metrics_names,results.tolist()))    
    crossval_df = pd.DataFrame(results.tolist()) 
    crossval_df.to_csv(f"./crossvalidation_results/{config.epochs}Epochs_{config.activation}Activation-{config.optimizer}Optimizer-{config.augmentation}Augmentation_lrate{config.learningrate}_mom{config.momentum}-crossvalidation_results.csv")  

        

