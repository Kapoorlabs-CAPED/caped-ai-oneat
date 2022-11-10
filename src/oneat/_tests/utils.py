
import numpy as np
import os
from pathlib import Path 
from oneat.NEATUtils.MovieCreator import train_test_split

def random_image_4d(shape = (3,8,64,64)):
    
    if type(shape) is list:
        shape = tuple(shape)
    image = np.random.rand(*shape)
    np.zeros
    return image 


def _root_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _create_train_val_data():
    
     num_train = 10
     train_size = 0.9
     #Generate 10 training images
     data = [random_image_4d() for _ in range(num_train)]    
     label = [(0,1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1) for _ in range(num_train)]
     dataarr = np.asarray(data)
     labelarr = np.asarray(label)
     traindata, validdata, trainlabel, validlabel = train_test_split(
        dataarr,
        labelarr,
        train_size= train_size,
        test_size=1 - train_size,
        shuffle=True,
     )
     
     return traindata, validdata, trainlabel, validlabel
     
    

    

def path_model_voll():
    return Path(_root_dir()) / '..' / 'models' / 'test'