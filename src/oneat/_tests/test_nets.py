import pytest
from .utils import random_image_4d
from oneat.NEATModels import NEATDenseNet, config

@pytest.mark.parametrize('img', (random_image_4d(3,8,64,64)))
def test_densent_input(img):
    
    conf = config()
