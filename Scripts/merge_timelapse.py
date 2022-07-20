from tifffile import imread, imwrite
from pathlib import Path
import numpy as np
import os
import glob
from natsort import natsorted
imagedir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/test/raw/third_dataset_split/seg/StarDist/'
savedir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/test/seg/'
Name = 'test_third_dataset_star'

Path(savedir).mkdir(exist_ok=True)
Raw_path = os.path.join(imagedir, '*tif')
filesRaw = glob.glob(Raw_path)
filesRaw = natsorted(filesRaw)

allseg = []               
for fname in filesRaw:
                segimage = imread(fname).astype('uint16')
            
                allseg.append(segimage)

allseg = np.asarray(allseg)
imwrite(savedir + '/' + Name +  '.tif', allseg.astype('uint16') )
