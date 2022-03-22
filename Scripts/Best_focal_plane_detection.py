#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
import glob
from oneat.NEATModels import NEATFocus
from oneat.NEATUtils.helpers import load_json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path


# In[ ]:


imagedir = '/Z_raw/'
model_dir = '/Oneat_models/'
savedir= '/Save_dir/'

model_name = 'cadhistonefocus'
focus_categories_json = model_dir + 'FocusCategories.json'
catconfig = load_json(focus_categories_json)
focus_cord_json = model_dir + 'FocusCord.json'
cordconfig = load_json(focus_cord_json)
model = NEATFocus(None, model_dir , model_name,catconfig, cordconfig)
Path(savedir).mkdir(exist_ok=True)
n_tiles = (1,1)


# # In the code block below compute the markers and make a dictionary for each image

# In[ ]:


Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)


for imagename in X:
     
         model.predict(imagename, 
                       savedir, 
                       n_tiles = n_tiles)

