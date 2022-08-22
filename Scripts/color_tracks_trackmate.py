#!/usr/bin/env python

import numpy as np
import os
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
import napari
import napatrackmater.bTrackmate as TM
from pathlib import Path
import qtpy
qtpy.QtWidgets.QApplication([])

# In[2]:


#Trackmate writes an XML file of tracks, we use it as input
xml_path = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/tracking_results/tracks_third_dataset_star_mari_principle_a30.xml' 
#Path to Segmentation image for extracting any track information from labels 
LabelImage = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/seg/Max_third_dataset_star.tif'
#Trackmate writes a spots and tracks file as csv
spot_csv = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/tracking_results/spots_third_dataset.csv'
track_csv = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/tracking_results/tracks_third_dataset.csv'
savedir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/tracking_results'
Path(savedir).mkdir(exist_ok=True)
scale = 255


# In[3]:


TM.import_TM_XML_Relabel(xml_path,LabelImage,spot_csv, track_csv, savedir, scale = scale)


# In[ ]:





# In[ ]:




