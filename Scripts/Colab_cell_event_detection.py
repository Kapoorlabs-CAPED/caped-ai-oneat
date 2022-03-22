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


# In[2]:



import sys
import os
import glob
from oneat.NEATModels import NEATStatic, nets
from oneat.NEATModels.Staticconfig import static_config
from oneat.NEATUtils.helpers import load_json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path


# In[ ]:


imagedir = '/Static_raw/'
model_dir = '/Oneat_models/'
savedir= '/Save_dir/'
model_name = 'CellNet'
division_categories_json = model_dir + 'StaticCategories.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'StaticCord.json'
cordconfig = load_json(division_cord_json)
model = NEATStatic(None, model_dir , model_name,catconfig, cordconfig)
Path(savedir).mkdir(exist_ok=True)
n_tiles = (2,2)
event_threshold = 1.0 - 1.0E-4
iou_threshold = 0.1


# In[ ]:


Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)
for imagename in X:
     model.predict(imagename, 
                   savedir, 
                   n_tiles = n_tiles, 
                   event_threshold = event_threshold, 
                   iou_threshold = iou_threshold)

