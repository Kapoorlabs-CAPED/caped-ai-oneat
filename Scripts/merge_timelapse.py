from tifffile import imread, imwrite
from pathlib import Path
import numpy as np
import os
import glob
from natsort import natsorted
import re
imagedir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/raw/third_dataset_split/seg/StarDist/'
savedir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/seg/'
search_pattern = '[0-9]+$'



Path(savedir).mkdir(exist_ok=True)
Raw_path = os.path.join(imagedir, '*tif')
filesRaw = glob.glob(Raw_path)
filesRaw = natsorted(filesRaw)

allseg = []               
for fname in filesRaw:
                segimage = imread(fname).astype('uint16')
            
                allseg.append(segimage)


search_output = re.search(search_pattern,fname)
save_name = fname[0:search_output.start()]
allseg = np.asarray(allseg)
imwrite(savedir + '/' + save_name +  '.tif', allseg.astype('uint16') )
