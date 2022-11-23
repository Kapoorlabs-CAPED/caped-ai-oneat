import pytest
from .utils import random_image_4d, create_train_val_data, root_dir
from oneat.NEATModels import NEATDenseVollNet, config

@pytest.mark.parametrize('shape', [(8,64,64,3), (8,32,32,3)])
@pytest.mark.parametrize('depth', [{'depth_0' :1, 'depth_2' :1, 'depth_3' :1}])
@pytest.mark.parametrize('stage_number', [3])
def test_densent_input(shape, depth, stage_number):
    
    img = random_image_4d(shape)
    catconfig = {'Normal': 0, 'Division': 1}
    tmpdir = root_dir()
    cordconfig = {"x": 0, "y": 1, "z": 2, "t": 3, "h": 4, "w": 5, "d": 6, "c": 7}
    params = config.volume_config(key_categories = catconfig, key_cord = cordconfig, epochs = 0, depth = depth, stage_number = stage_number)
    oneat_model = NEATDenseVollNet(params, str(tmpdir), catconfig = catconfig, cordconfig = cordconfig )
    
    oneat_model.X,oneat_model.X_val,oneat_model.Y,oneat_model.Y_val = create_train_val_data(img)
    
    oneat_model.Y = oneat_model.Y.reshape((oneat_model.Y.shape[0],1, 1, 1, oneat_model.Y.shape[1]))
    oneat_model.Y_val = oneat_model.Y_val.reshape((oneat_model.Y_val.shape[0],1, 1, 1, oneat_model.Y_val.shape[1]))
    oneat_model.TrainModel()
    keras_model = oneat_model.Trainingmodel
    assert keras_model.layers[1].output_shape[-1] == oneat_model.startfilter
    assert keras_model.layers[1].output_shape[1] is None
    assert keras_model.layers[1].output_shape[2] is None
    assert keras_model.layers[1].output_shape[3] is None
    
    assert keras_model.layers[-1].output_shape[-1] == len(catconfig) + len(cordconfig)
    assert keras_model.layers[-1].output_shape[1] is None
    assert keras_model.layers[-1].output_shape[2] is None
    assert keras_model.layers[-1].output_shape[3] is None
    
    assert oneat_model.yolo_loss.__name__ == 'loss'
    
    
