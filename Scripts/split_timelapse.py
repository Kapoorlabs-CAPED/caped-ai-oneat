from tifffile import imread, imwrite
from pathlib import Path
import numpy as np
import os
import glob
from natsort import natsorted
imagedir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data/Dataset3/'
savedir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data/split_Dataset3/'
Path(savedir).mkdir(exist_ok=True)
Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)
X = natsorted(X)

for imagename in X:
                print(imagename)
                image = imread(imagename)
                Name = os.path.basename(os.path.splitext(imagename)[0])
                for i in range(image.shape[0]):
                    imwrite(savedir + '/' + Name + str(i) +  '.tif', image[i,:,:,:].astype('float32') )
               
