

import sys
import os
import glob
from oneat.NEATModels import NEATSynamic
from oneat.NEATModels.config import dynamic_config
from oneat.NEATUtils import helpers
from oneat.NEATUtils.helpers import load_json
from pathlib import Path


# In[2]:


n_tiles = (1,1)
event_threshold = 0.999
event_confidence = 0.9
iou_threshold = 0.3
fidelity = 4
downsamplefactor = 1
nms_function = 'iou'

# In[3]:


imagedir = '/gpfsstore/rech/jsy/uzj81mi/Oneat_Data/TestDatasets/Hela_flou_oneat/mcherry/'
model_dir = '/gpfsstore/rech/jsy/uzj81mi/Oneat_Data/h5_json_files/'
savedir= '/gpfsstore/rech/jsy/uzj81mi/Oneat_Data/TestDatasets/Hela_flou_oneat/mcherry/Results/'
model_name = 'Cellsplitdetectormcherry'

division_categories_json = model_dir + 'Cellsplitcategorieshelaflou.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'Cellsplitcordhelaflou.json'
cordconfig = load_json(division_cord_json)
model = NEATSynamic(None, model_dir , model_name,catconfig, cordconfig)
Path(savedir).mkdir(exist_ok=True)

Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)

for imagename in X:

     model.predict_synamic(imagename,
                           savedir,
                           n_tiles = n_tiles,
                           event_threshold = event_threshold,
                           event_confidence = event_confidence,
                           iou_threshold = iou_threshold, 
                           fidelity = fidelity,
                           nms_function = nms_function,
                           downsamplefactor = downsamplefactor )