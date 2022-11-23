import pytest
from .utils import random_image_4d, create_train_val_data, root_dir
from oneat.NEATModels import NEATDenseVollNet, config
import keras

def test_densent_input():
    
    img = random_image_4d(shape = (3,8,64,64))
    catconfig = {'Normal': 0, 'Division': 1}
    tmpdir = root_dir()
    cordconfig = {"x": 0, "y": 1, "z": 2, "t": 3, "h": 4, "w": 5, "d": 6, "c": 7}
    params = config.volume_config(key_categories = catconfig, key_cord = cordconfig, epochs = 1, depth = {'depth_0' :1, 'depth_2' :1, 'depth_3' :1})
    oneat_model = NEATDenseVollNet(params, str(tmpdir), catconfig = catconfig, cordconfig = cordconfig )
    
    oneat_model.X,oneat_model.X_val,oneat_model.Y,oneat_model.Y_val = create_train_val_data(img)
    
    oneat_model.Y = oneat_model.Y.reshape((oneat_model.Y.shape[0],1, 1, 1, oneat_model.Y.shape[1]))
    oneat_model.Y_val = oneat_model.Y_val.reshape((oneat_model.Y_val.shape[0],1, 1, 1, oneat_model.Y_val.shape[1]))
    oneat_model.TrainModel()
    assert oneat_model.yolo_loss.__name__ == 'loss'
    
    
