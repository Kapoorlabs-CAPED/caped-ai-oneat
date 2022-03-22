#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive', force_remount = True)
get_ipython().run_line_magic('tensorflow_version', '2.x')


# In[ ]:


get_ipython().system('pip uninstall keras -y')
get_ipython().system('pip uninstall keras-nightly -y')
get_ipython().system('pip uninstall keras-Preprocessing -y')
get_ipython().system('pip uninstall keras-vis -y')
get_ipython().system('pip uninstall tensorflow -y')

get_ipython().system('pip install tensorflow==2.2.0')
get_ipython().system('pip install keras==2.3.0')
get_ipython().system('pip install oneat')


# In[ ]:


import sys
import os
from glob import glob
from oneat.NEATModels import NEATPredict
from oneat.NEATUtils.helpers import load_json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# In[ ]:


Z_imagedir = 'Z_raw/'
imagedir = 'Projected_raw/'
model_dir =  '/Oneat_models/'
model_name = 'Cellsplitpredictor'

division_categories_json = model_dir + 'MicroscopeCategories.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'MicroscopeCord.json'
cordconfig = load_json(division_cord_json)
fileextension = '*TIF'

model = NEATPredict(None, model_dir , model_name,catconfig, cordconfig)
projection_model = ProjectionCARE(config = None, name = projection_model_name, basedir = model_dir)


# In[ ]:


n_tiles = (1,1)
Z_n_tiles = (1,2,2)
event_threshold = 1.0 - 1.0E-4
iou_threshold = 0.1
nb_predictions = 30
fidelity = 8
downsample = 1


# In[ ]:


model.predict_microscope(imagedir, 
              [], 
              [], 
              Z_imagedir, 
              [], 
              [],  
              0, 
              0,
              downsample = downsample,
              fileextension = fileextension, 
              nb_prediction = nb_predictions, 
              n_tiles = n_tiles, 
              Z_n_tiles = Z_n_tiles, 
              event_threshold = event_threshold, 
              iou_threshold = iou_threshold,
              fidelity = fidelity)

