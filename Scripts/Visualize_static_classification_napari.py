#!/usr/bin/env python
# coding: utf-8

# In[9]:



import sys
import os
import glob
from oneat.NEATUtils import NEATViz


get_ipython().run_line_magic('gui', 'qt')


# In[10]:


categories_json = '/home/sancere/VKepler/CurieDeepLearningModels/OneatModels/MasterBinning2V1Models/DynamicCategories.json'


# In[11]:


imagedir = '/home/sancere/VKepler/oneatgolddust/Test/Bin2Test/'
csvdir = '/home/sancere/VKepler/oneatgolddust/Test/Bin2Test/Save29resf16/'
fileextension = '*tif'
event_threshold = [1, 0.99999, 0.9999999, 0.9, 0.9, 0.9]
Vizdetections = NEATViz(imagedir, csvdir, categories_json, event_threshold, fileextension = fileextension)


# In[12]:


imagedir = '/home/sancere/VKepler/oneatgolddust/Test/Bin2Test/'
csvdir = '/home/sancere/VKepler/oneatgolddust/Test/Bin2Test/FNMSHTDivd29resf32/'
fileextension = '*tif'
thresh = 1.0 - 1.0E-5
event_threshold = [thresh, thresh, thresh, thresh, thresh, thresh]
Vizdetections = NEATViz(imagedir, csvdir, categories_json, event_threshold, fileextension = fileextension)


# In[13]:



imagedir ='/home/sancere/VKepler/oneatgolddust/Test/Bin2Test/'
csvdir = '/home/sancere/VKepler/oneatgolddust/Test/Bin2Test/FNMSHTMicrod29resf32/'

fileextension = '*tif'
thresh = 1.0 - 1.0E-5
event_threshold = [thresh, thresh, thresh, thresh, thresh, thresh]
Vizdetections = NEATViz(imagedir, csvdir, categories_json, event_threshold, fileextension = fileextension)


# In[14]:



imagedir ='/home/sancere/VKepler/oneatgolddust/Test/Bin2Test/'
csvdir = '/home/sancere/VKepler/oneatgolddust/Test/Bin2Test/longmasterd29resf32/'

fileextension = '*tif'
event_threshold = [1, 0.9999, 0.999999, 0.9, 0.9, 0.9]

Vizdetections = NEATViz(imagedir, csvdir, categories_json, event_threshold, fileextension = fileextension)


# In[15]:



imagedir ='/home/sancere/VKepler/oneatgolddust/Test/Bin2Test/'
csvdir = '/home/sancere/VKepler/oneatgolddust/Test/Bin2Test/MasterMicrod29f32/'

fileextension = '*tif'
event_threshold = [1, 0.9999, 0.999999, 0.9, 0.9, 0.9]

Vizdetections = NEATViz(imagedir, csvdir, categories_json, event_threshold, fileextension = fileextension)


# In[ ]:




