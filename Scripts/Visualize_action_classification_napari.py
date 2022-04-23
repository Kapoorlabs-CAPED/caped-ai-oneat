#!/usr/bin/env python
# coding: utf-8

# In[17]:

from oneat.NEATUtils import NEATViz




# In[19]:


imagedir = 'D:/TestDatasets/Oneat/Hela_flou/'
heatmapdir ='D:/TestDatasets/Oneat/Hela_flou/results_d29f32s3/'
csvdir = 'D:/TestDatasets/Oneat/Hela_flou/results_d29f32s3/'
categories_json = 'D:/TrainingModels/Oneat/Cellsplithelafloucategories.json'
fileextension = '*tif'
thresh = 0.99999
event_threshold = [thresh, thresh]
Vizdetections = NEATViz(imagedir, None, csvdir, categories_json, event_threshold, fileextension = fileextension)


# In[ ]:





# In[ ]:




