from .NEATModels import *
from .NEATUtils import *
from .version import __version__
from csbdeep.utils.tf import keras_import
from . import NEATDynamic, NEATSynamic

from oneat.pretrained import register_model, register_aliases, clear_models_and_aliases

get_file = keras_import('utils', 'get_file')


clear_models_and_aliases(NEATDynamic, NEATSynamic)

register_model(NEATSynamic,   'brightfield_mitosis_oneat_model',  'https://zenodo.org/record/6481021/files/Cellsplitdetectorbrightfield.h5', '0c6ba49c1ba0eb91819af40460fb66cb'
                                                               , 'https://zenodo.org/record/6481021/files/Cellsplitcordhelaflou.json', 'aed21cb69d6fb8be32c47f78a39d32f5'
                                                                , 'https://zenodo.org/record/6481021/files/Cellsplitcategorieshelaflou.json', '7a67a83f08fb1add3c1b1a3e0eeec773'
                                                                , 'https://zenodo.org/record/6481021/files/Cellsplitdetectorbrightfield_Parameter.json', '266e3e7fa7587d53d62ae0492482a1bc'  )  

register_aliases(NEATSynamic, 'brightfield_mitosis_oneat_model',  'brightfield_mitosis_oneat_model')




del register_model, register_aliases, clear_models_and_aliases


def abspath(path):
    import os
    base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, path)



def test_image_brightfield():
    from tifffile import imread
    url = "https://zenodo.org/record/6480142/files/20210904_TL2%20-%20R05-C03-F0_ch_2.tif"
    hash = "67e13fa4df301dfe2c2a57f785aedada"
    
    img = imread(abspath(get_file(fname='brightfield', origin=url, file_hash=hash)))
    return img
