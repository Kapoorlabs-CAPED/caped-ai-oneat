from tifffile import imread, imwrite
from pathlib import Path
import numpy as np
import os
import glob
from natsort import natsorted
imagedir = '/gpfsstore/rech/jsy/uzj81mi/Tracking_challenge/'
savedir = '/gpfsstore/rech/jsy/uzj81mi/Tracking_challenge/split_dataset/'
Path(savedir).mkdir(exist_ok=True)
Raw_path = os.path.join(imagedir, '*tif')
filesRaw = glob.glob(Raw_path)
filesRaw = natsorted(filesRaw)

for imagename in filesRaw:
                print(imagename)
                image = imread(imagename)
                Name = os.path.basename(os.path.splitext(imagename)[0])
                for i in range(image.shape[0]):
                    imwrite(savedir + '/' + Name + str(i)  +  '.tif', image[i,:,:,:].astype('uint16') )
               
