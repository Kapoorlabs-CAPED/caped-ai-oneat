import sys
import os
import glob
from oneat.NEATModels import NEATFocus, nets
from oneat.NEATModels.config import dynamic_config
from oneat.NEATUtils import helpers
from oneat.NEATUtils.helpers import load_json
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path




imagedir = 'images/'
model_dir = 'models/'
savedir= 'results/'

model_name = 'Focalplanedetector'
focus_categories_json = model_dir + 'Focalplanecategories.json'
catconfig = load_json(focus_categories_json)
focus_cord_json = model_dir + 'Focalplanecord.json'
cordconfig = load_json(focus_cord_json)
model = NEATFocus(None, model_dir , model_name,catconfig, cordconfig)
Path(savedir).mkdir(exist_ok=True)
n_tiles = (1,1)
interest_event = ("BestCad", "BestNuclei")




Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)


for imagename in X:
     
         model.predict(imagename, savedir, interest_event, n_tiles = n_tiles)






