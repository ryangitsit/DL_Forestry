    
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

def test():
    filenames = []
    labels = []
      

    for file in os.listdir('PetImages/Cat'):
        filenames.append(os.path.join('Cat', file))
        labels.append('Cat')

    for file in os.listdir('PetImages/Dog'):
        filenames.append(os.path.join('Dog', file))
        labels.append('Dog')


    d = {'filename': filenames, 'label': labels}
    alldata = pd.DataFrame(d)
    print(alldata.head())
    alldata = alldata.sample(frac=1).reset_index(drop=True) 
    Y = alldata[['label']]
    print(alldata.head())


