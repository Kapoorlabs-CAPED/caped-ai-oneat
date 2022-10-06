import numpy as np
from oneat.NEATUtils.utils import load_json, normalizeFloatZeroOne,  focyoloprediction, simpleaveragenms
import os
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from tifffile import imwrite
import time
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from oneat.NEATModels.neat_focus import NEATFocus
from oneat.NEATModels.nets import Concat
from keras import backend as K
#from IPython.display import clear_output
from pathlib import Path
from keras.models import load_model
from tifffile import imread
import csv
from scipy.optimize import curve_fit
from scipy.ndimage import zoom
import glob
#from napari.qt.threading import thread_worker
#from matplotlib.backends.backend_qt5agg import \
    #FigureCanvasQTAgg as FigureCanvas
#from qtpy.QtCore import Qt
#from qtpy.QtWidgets import QComboBox, QPushButton, QSlider
import h5py
Boxname = 'ImageIDBox'
EventBoxname = 'EventIDBox'

class NEATFocusPredict(NEATFocus):
    
    def __init__(self, config, model_dir, model_name, catconfig, cordconfig):

          super().__init__(config = config, model_dir = model_dir, model_name = model_name, catconfig = catconfig, cordconfig = cordconfig)



    def predict(self, imagedir, Z_imagedir, Z_movie_name_list, Z_movie_input, start,
                Z_start, downsample=False, fileextension='*TIF', nb_prediction=3, Z_n_tiles=(1, 2, 2),
                overlap_percent=0.6, normalize = True):

        self.imagedir = imagedir
        self.basedirResults = self.imagedir + '/' + "live_results"
        Path(self.basedirResults).mkdir(exist_ok=True)
        # Recurrsion variables
        self.Z_movie_name_list = Z_movie_name_list
        self.Z_movie_input = Z_movie_input
        self.Z_imagedir = Z_imagedir
        self.start = start
        self.Z_start = Z_start
        self.nb_prediction = nb_prediction
        self.fileextension = fileextension
        self.Z_n_tiles = Z_n_tiles
        self.overlap_percent = overlap_percent
        self.downsample = downsample
        self.normalize = normalize
        f = h5py.File(self.model_dir + self.model_name + '.h5', 'r+')
        data_p = f.attrs['training_config']
        data_p = data_p.decode().replace("learning_rate", "lr").encode()
        f.attrs['training_config'] = data_p
        f.close()
        self.model = load_model(self.model_dir + self.model_name + '.h5',
                                custom_objects={'loss': self.yolo_loss, 'Concat': Concat})

        # Z slice folder listener
        while 1:

            Z_Raw_path = os.path.join(self.Z_imagedir, self.fileextension)
            Z_filesRaw = glob.glob(Z_Raw_path)

            for Z_movie_name in Z_filesRaw:
                Z_Name = os.path.basename(os.path.splitext(Z_movie_name)[0])
                # Check for unique filename
                if Z_Name not in self.Z_movie_name_list:
                    self.Z_movie_name_list.append(Z_Name)
                    self.Z_movie_input.append(Z_movie_name)

                    if Z_Name in self.Z_movie_name_list:
                            self.Z_movie_name_list.remove(Z_Name)
                    if Z_movie_name in self.Z_movie_input:
                            self.Z_movie_input.remove(Z_movie_name)

            self.Z_movie_input_list = []
            for (k, v) in self.Z_movie_input.items():
                self.Z_movie_input_list.append(v)
            total_movies = len(self.Z_movie_input_list)

            if total_movies > self.start:
                current_movies = imread(self.Z_movie_input_list[self.start:self.start + 1])

                sizey = current_movies.shape[0]
                sizex = current_movies.shape[1]
                if self.downsample:
                    scale_percent = 50
                    width = int(sizey * scale_percent / 100)
                    height = int(sizex * scale_percent / 100)
                    dim = (width, height)
                    sizex = height
                    sizey = width

                    current_movies_down = np.zeros([sizey, sizex])
                    # resize image
                    current_movies_down = zoom.resize(current_movies, dim)
                else:
                    current_movies_down = current_movies
                # print(current_movies_down.shape)
                print('Predicting on Movie:', self.Z_movie_input_list[self.start:self.start + 1])
                inputtime = self.start

                eventboxes = []
                classedboxes = {}
                self.image = current_movies_down
                if self.normalize:
                    self.image = normalizeFloatZeroOne(self.image, 1, 99.8)
                # Break image into tiles if neccessary

                print('Doing ONEAT prediction')
                start_time = time.time()



                # Iterate over tiles

                for inputz in tqdm(range(0, self.image.shape[0])):
                    if inputz <= self.image.shape[0] - self.imagez:

                        eventboxes = []
                        classedboxes = {}
                        smallimage = CreateVolume(self.image, self.imagez, inputz)
                        predictions, allx, ally = self.predict_main(smallimage)
                        for p in range(0, len(predictions)):

                            sum_z_prediction = predictions[p]

                            if sum_z_prediction is not None:
                                # For each tile the prediction vector has shape N H W Categories + Training Vector labels
                                for i in range(0, sum_z_prediction.shape[0]):
                                    z_prediction = sum_z_prediction[i]
                                    boxprediction = focyoloprediction(ally[p], allx[p], z_prediction, self.stride, inputz,
                                                                      self.config, self.key_categories)

                                    if boxprediction is not None:
                                        eventboxes = eventboxes + boxprediction

                        for (event_name, event_label) in self.key_categories.items():

                            if event_label > 0:
                                current_event_box = []
                                for box in eventboxes:

                                    event_prob = box[event_name]
                                    if event_prob > 0 :
                                        current_event_box.append(box)
                                classedboxes[event_name] = [current_event_box]

                        self.classedboxes = classedboxes
                        self.eventboxes = eventboxes

                        self.nms()
                        self.to_csv()
                        self.draw()

                print("____ Prediction took %s seconds ____ ", (time.time() - start_time))
                self.print_planes()
                self.genmap()
                self.start = self.start + 1
                self.predict(self.imagedir,  self.Z_imagedir,
                             self.Z_movie_name_list, self.Z_movie_input, self.start, Z_start,
                             fileextension=self.fileextension, downsample=self.downsample,
                             nb_prediction=self.nb_prediction,  Z_n_tiles=self.Z_n_tiles,
                             overlap_percent=self.overlap_percent)

    def nms(self):

        best_iou_classedboxes = {}
        all_best_iou_classedboxes = {}
        self.all_iou_classedboxes = {}
        self.iou_classedboxes = {}
        for (event_name, event_label) in self.key_categories.items():
            if event_label > 0:
                # Get all events

                sorted_event_box = self.classedboxes[event_name][0]

                sorted_event_box = sorted(sorted_event_box, key=lambda x: x[event_name], reverse=True)

                scores = [sorted_event_box[i][event_name] for i in range(len(sorted_event_box))]
                best_sorted_event_box, all_boxes = simpleaveragenms(sorted_event_box, scores, self.iou_threshold,
                                                                    self.event_threshold, event_name)

                all_best_iou_classedboxes[event_name] = [all_boxes]
                best_iou_classedboxes[event_name] = [best_sorted_event_box]
        self.iou_classedboxes = best_iou_classedboxes
        self.all_iou_classedboxes = all_best_iou_classedboxes

    def genmap(self):

        image = imread(self.savename)
        Name = os.path.basename(os.path.splitext(self.savename)[0])
        Signal_first = image[:, :, :, 1]
        Signal_second = image[:, :, :, 2]
        Sum_signal_first = gaussian_filter(np.sum(Signal_first, axis=0), self.radius)
        Sum_signal_first = normalizeZeroOne(Sum_signal_first)
        Sum_signal_second = gaussian_filter(np.sum(Signal_second, axis=0), self.radius)

        Sum_signal_second = normalizeZeroOne(Sum_signal_second)

        Zmap = np.zeros([Sum_signal_first.shape[0], Sum_signal_first.shape[1], 3])
        Zmap[:, :, 0] = Sum_signal_first
        Zmap[:, :, 1] = Sum_signal_second
        Zmap[:, :, 2] = (Sum_signal_first + Sum_signal_second) / 2

        imwrite(self.basedirResults + Name + '_Zmap' + '.tif', Zmap)

    def to_csv(self):
        

        
        for (event_name,event_label) in self.key_categories.items():
                   
            
                   
                   if event_label > 0:
                                            zlocations = []
                                            scores = []
                                            max_scores = []
                                            iou_current_event_box = self.iou_classedboxes[event_name][0]
                                            zcenter = iou_current_event_box['real_z_event']
                                            max_score = iou_current_event_box['max_score']
                                            score = iou_current_event_box[event_name]
                                                   
                                            zlocations.append(zcenter)
                                            scores.append(score)
                                            max_scores.append(max_score)
                                            print(zlocations, scores)
                                            event_count = np.column_stack([zlocations,scores, max_scores]) 
                                            event_count = sorted(event_count, key = lambda x:x[0], reverse = False)
                                            event_data = []
                                            csvname = self.basedirResults + "/" + (os.path.splitext(os.path.basename(self.imagename))[0])  + event_name  +  "_FocusQuality"
                                            writer = csv.writer(open(csvname  +".csv", "a"))
                                            filesize = os.stat(csvname + ".csv").st_size
                                            if filesize < 1:
                                               writer.writerow(['Z','Score','Max_score'])
                                            for line in event_count:
                                               if line not in event_data:  
                                                  event_data.append(line)
                                               writer.writerows(event_data)
                                               event_data = []           
                              
                                              
                                            
                                              
    
    def fit_curve(self):


                                   for (event_name,event_label) in self.key_categories.items():



                                         if event_label > 0:         
                                              readcsvname = self.basedirResults + "/" + (os.path.splitext(os.path.basename(self.imagename))[0])  + event_name  +  "_FocusQuality"
                                              self.dataset   = pd.read_csv(readcsvname, delimiter = ',')
                                              self.dataset_index = self.dataset.index
            
            
                                              Z = self.dataset[self.dataset.keys()[0]][1:]
                                              score = self.dataset[self.dataset.keys()[1]][1:]
                                              
                                              H, A, mu0, sigma = gauss_fit(np.array(Z), np.array(score))
                                              csvname = self.basedirResults + "/" + (os.path.splitext(os.path.basename(self.imagename))[0])  + event_name  +  "_GaussFitFocusQuality"
                                              writer = csv.writer(open(csvname  +".csv", "a"))
                                              filesize = os.stat(csvname + ".csv").st_size
                                              if filesize < 1:
                                                 writer.writerow(['Amplitude','Mean','Sigma'])
                                                 writer.writerow([A, mu0,sigma])
                                              
                                              csvname = self.basedirResults + "/" + event_name

                                                
                                                 


    def print_planes(self):
        for (event_name,event_label) in self.key_categories.items():
             if event_label > 0:
                     csvfname =  self.basedirResults + "/" + (os.path.splitext(os.path.basename(self.imagename))[0])  + event_name  +  "_FocusQuality" + ".csv"
                     dataset = pd.read_csv(csvfname, skiprows = 0)
                     z = dataset[dataset.keys()[0]][1:]
                     score = dataset[dataset.keys()[1]][1:]
                     terminalZ = dataset[dataset.keys()[2]][1:]
                     subZ = terminalZ[terminalZ > 0.1]
                     maxscore = np.max(score)
                     try: 
                         maxz = z[np.argmax(score)] + 2
                       

                         print('Best Zs'+ (os.path.splitext(os.path.basename(self.imagename))[0]) + 'for'+ event_name + 'at' +  str(maxz))
                     except:

                           pass


    def draw(self):
         
          for (event_name,event_label) in self.key_categories.items():
                   
                  if event_label > 0:
                                  
                                   xlocations = []
                                   ylocations = []
                                   scores = []
                                   zlocations = []   
                                   heights = []
                                   widths = [] 
                                   iou_current_event_boxes = self.all_iou_classedboxes[event_name][0]
                                   
                                  
                                                                           
                                   for iou_current_event_box in iou_current_event_boxes:
                                              
                                             
                                              xcenter = iou_current_event_box['xcenter']
                                              ycenter = iou_current_event_box['ycenter']
                                              zcenter = iou_current_event_box['real_z_event']
                                              xstart = iou_current_event_box['xstart']
                                              ystart = iou_current_event_box['ystart']
                                              xend = xstart + iou_current_event_box['width']
                                              yend = ystart + iou_current_event_box['height']
                                              score = iou_current_event_box[event_name]

                                              
                                              
                                                            
                                                            
                                              if event_label == 1:
                                                  for x in range(int(xstart),int(xend)):
                                                      for y in range(int(ystart), int(yend)):
                                                                if y < self.image.shape[1] and x < self.image.shape[2]:
                                                                    self.Maskimage[int(zcenter), y, x, 1] = self.Maskimage[int(zcenter), y, x, 1] + score
                                              else:
                                                  
                                                  for x in range(int(xstart),int(xend)):
                                                      for y in range(int(ystart), int(yend)):
                                                          if y < self.image.shape[1] and x < self.image.shape[2]:
                                                              self.Maskimage[int(zcenter), y, x, 2] = self.Maskimage[int(zcenter), y, x, 2] +  score
                                            
                                                  
                                                  
                                              if score > 0.9:
                                                  
                                                 xlocations.append(round(xcenter))
                                                 ylocations.append(round(ycenter))
                                                 scores.append(score)
                                                 zlocations.append(zcenter)
                                                 heights.append(iou_current_event_box['height'])
                                                 widths.append(iou_current_event_box['width'] )  
        
                                   
    def overlaptiles(self, sliceregion):
        
             if self.n_tiles == (1, 1):
                               patch = []
                               rowout = []
                               column = []
                               patchx = sliceregion.shape[2] // self.n_tiles[0]
                               patchy = sliceregion.shape[1] // self.n_tiles[1]
                               patchshape = (patchy, patchx) 
                               smallpatch, smallrowout, smallcolumn =  chunk_list(sliceregion, patchshape, self.stride, [0,0])
                               patch.append(smallpatch)
                               rowout.append(smallrowout)
                               column.append(smallcolumn)
                 
             else:     
                     patchx = sliceregion.shape[2] // self.n_tiles[0]
                     patchy = sliceregion.shape[1] // self.n_tiles[1]
                
                     if patchx > self.imagex and patchy > self.imagey:
                          if self.overlap_percent > 1 or self.overlap_percent < 0:
                             self.overlap_percent = 0.8
                         
                          jumpx = int(self.overlap_percent * patchx)
                          jumpy = int(self.overlap_percent * patchy)
                         
                          patchshape = (patchy, patchx)   
                          rowstart = 0; colstart = 0
                          pairs = []  
                          #row is y, col is x
                          
                          while rowstart < sliceregion.shape[1] -patchy:
                             colstart = 0
                             while colstart < sliceregion.shape[2] -patchx:
                                
                                 # Start iterating over the tile with jumps = stride of the fully convolutional network.
                                 pairs.append([rowstart, colstart])
                                 colstart+=jumpx
                             rowstart+=jumpy 
                            
                          #Include the last patch   
                          rowstart = sliceregion.shape[1] -patchy
                          colstart = 0
                          while colstart < sliceregion.shape[2]:
                                        pairs.append([rowstart, colstart])
                                        colstart+=jumpx
                          rowstart = 0
                          colstart = sliceregion.shape[2] -patchx
                          while rowstart < sliceregion.shape[1]:
                                        pairs.append([rowstart, colstart])
                                        rowstart+=jumpy              
                                        
                          if sliceregion.shape[1] >= self.imagey and sliceregion.shape[2]>= self.imagex :          
                                
                                patch = []
                                rowout = []
                                column = []
                                for pair in pairs: 
                                   smallpatch, smallrowout, smallcolumn =  chunk_list(sliceregion, patchshape, self.stride, pair)
                                   if smallpatch.shape[1] >= self.imagey and smallpatch.shape[2] >= self.imagex: 
                                           patch.append(smallpatch)
                                           rowout.append(smallrowout)
                                           column.append(smallcolumn) 
                        
                     else:
                         
                               patch = []
                               rowout = []
                               column = []
                               patchx = sliceregion.shape[2] // self.n_tiles[0]
                               patchy = sliceregion.shape[1] // self.n_tiles[1]
                               patchshape = (patchy, patchx)
                               smallpatch, smallrowout, smallcolumn =  chunk_list(sliceregion, patchshape, self.stride, [0,0])
                               patch.append(smallpatch)
                               rowout.append(smallrowout)
                               column.append(smallcolumn)
             self.patch = patch          
             self.sy = rowout
             self.sx = column            
          
    

    
    def predict_main(self,sliceregion):
            try:
                self.overlaptiles(sliceregion)
                predictions = []
                allx = []
                ally = []
                if len(self.patch) > 0:
                    for i in range(0,len(self.patch)):   
                               
                               sum_time_prediction = self.make_patches(self.patch[i])
                               predictions.append(sum_time_prediction)
                               allx.append(self.sx[i])
                               ally.append(self.sy[i])
                      
                       
                else:
                    
                       sum_time_prediction = self.make_patches(self.patch)
                       predictions.append(sum_time_prediction)
                       allx.append(self.sx)
                       ally.append(self.sy)
           
            except tf.errors.ResourceExhaustedError:
                
                print('Out of memory, increasing overlapping tiles for prediction')
                
                self.list_n_tiles = list(self.n_tiles)
                self.list_n_tiles[0] = self.n_tiles[0]  + 1
                self.list_n_tiles[1] = self.n_tiles[1]  + 1
                self.n_tiles = tuple(self.list_n_tiles) 
                
                
                self.predict_main(sliceregion)
                
            return predictions, allx, ally
        
    def make_patches(self, sliceregion):
       
       
       predict_im = np.expand_dims(sliceregion,0)
       
       
       prediction_vector = self.model.predict(np.expand_dims(predict_im,-1), verbose = 0)
         
            
       return prediction_vector
   
    def make_batch_patches(self, sliceregion): 
   
      
               prediction_vector = self.model.predict(np.expand_dims(sliceregion,-1), verbose = 0)
               return prediction_vector
         
            
          
               

        
def chunk_list(image, patchshape, stride, pair):
            rowstart = pair[0]
            colstart = pair[1]

            endrow = rowstart + patchshape[0]
            endcol = colstart + patchshape[1]

            if endrow > image.shape[1]:
                endrow = image.shape[1]
            if endcol > image.shape[2]:
                endcol = image.shape[2]


            region = (slice(0,image.shape[0]),slice(rowstart, endrow),
                      slice(colstart, endcol))

            # The actual pixels in that region.
            patch = image[region]

            # Always normalize patch that goes into the netowrk for getting a prediction score 
            


            return patch, rowstart, colstart
        
        
def CreateVolume(patch, imagez, timepoint):
    
               starttime = timepoint
               endtime = timepoint + imagez
               smallimg = patch[starttime:endtime, :]
       
               return smallimg         

def normalizeZeroOne(x):
    x = x.astype('float32')

    minVal = np.min(x)
    maxVal = np.max(x)

    x = ((x - minVal) / (maxVal - minVal + 1.0e-20))

    return x



    
def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y):
     
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
    return popt    
    
