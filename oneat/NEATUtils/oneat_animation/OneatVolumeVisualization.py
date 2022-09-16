from poplib import POP3_SSL_PORT
import pandas as pd
import numpy as np
import os
import napari
from skimage import measure
from dask.array.image import imread as daskread
from tifffile import imread
from skimage import morphology
class OneatVolumeVisualization:

    def __init__(self, viewer: napari.Viewer,key_categories, csvdir,
     savedir, savename, ax, figure):

        self.viewer = viewer
        self.csvdir = csvdir
        self.savedir = savedir
        self.savename = savename
        self.key_categories = key_categories
        self.ax = ax
        self.figure = figure
        self.dataset = None
        self.event_name = None
        self.cell_count = None      
        self.image = None
        self.seg_image = None
        self.event_locations = []
        self.event_locations_dict = {}
        self.event_locations_size_dict = {}
        self.size_locations = []
        self.score_locations = []
        self.confidence_locations = []
        self.event_locations_clean = [] 
        self.cleantimelist = []
        self.cleaneventlist= []
        self.cleannormeventlist = []
        self.cleancelllist = []
        self.labelsize = {}
        self.segimagedir  = None
        self.plot_event_name = None 
        self.event_count_plot = None
        self.event_norm_count_plot = None 
        self.cell_count_plot = None
        self.imagename = None
        self.originalimage = None
        
   
                
               

    def show_plot(self,  plot_event_name, event_count_plot, 
      segimagedir = None, event_threshold = 0 ):

        timelist = []
        eventlist= []
        normeventlist = []
        celllist = []
        self.ax.cla()
        
        self.segimagedir = segimagedir
        self.plot_event_name = plot_event_name
        self.event_count_plot = event_count_plot 
        
        if self.dataset is not None:                             
               
                for layer in list(self.viewer.layers):
                    if isinstance(layer, napari.layers.Image):
                            self.image = layer.data
                    if isinstance(layer, napari.layers.Labels):
                            self.seg_image = layer.data    


                if self.image is not None:    
                        currentT   = np.round(self.dataset["T"]).astype('int')
                        currentsize = self.dataset["Score"]
                            
                        for i in range(0, self.image.shape[0]):
                            
                            condition = currentT == i
                            condition_indices = self.dataset_index[condition]
                            conditionScore = currentsize[condition_indices]
                            score_condition = conditionScore > event_threshold
                            countT = len(conditionScore[score_condition])
                            timelist.append(i)
                            eventlist.append(countT)
                           
                        self.cleannormeventlist = []    
                        
                              
                        if self.plot_event_name == self.event_count_plot:    
                                self.ax.plot(timelist, eventlist, '-r')
                                self.ax.plot(self.cleantimelist, self.cleaneventlist, '-g')
                                self.ax.set_title(self.event_name + "Events")
                                self.ax.set_xlabel("Time")
                                self.ax.set_ylabel("Counts")
                                self.figure.canvas.draw()
                                self.figure.canvas.flush_events()
                                
                                self.figure.savefig(self.savedir  + self.event_name + self.event_count_plot + (os.path.splitext(os.path.basename(self.imagename))[0]  + '.png'), dpi = 300)

                       
                       

    def show_image(self, 
                image_toread, 
                imagename, 
                segimagedir = None, 
                heatmapimagedir = None, 
                heatname = '_Heat', 
                use_dask = False):
        self.imagename = imagename
        name_remove = ('Image', 'SegImage')
        for layer in list(self.viewer.layers):
                                         if  any(name in layer.name for name in name_remove):
                                                    self.viewer.layers.remove(layer)
        try:                                            
            if use_dask:                                      
                self.image = daskread(image_toread)[0]
            else:
                self.image = imread(image_toread)    
            
            if heatmapimagedir is not None:
                    try:
                        if use_dask: 
                            heat_image = daskread(heatmapimagedir + imagename + heatname + '.tif')[0]
                        else:
                            heat_image = imread(heatmapimagedir + imagename + heatname + '.tif')
                    except:
                        heat_image = None   
            
            if  segimagedir is not None:
                    if use_dask:
                        self.seg_image = daskread(segimagedir + imagename + '.tif')[0]
                    else:
                        self.seg_image = imread(segimagedir + imagename + '.tif')    

                      
                    
                    self.viewer.add_labels(self.seg_image.astype('uint16'), name = 'SegImage'+ imagename)
                   
                    
                        
            self.originalimage = self.image
            self.viewer.add_image(self.image, name= 'Image' + imagename )
            if heatmapimagedir is not None:
                    try:
                      self.viewer.add_image(heat_image, name= 'Image' + imagename + heatname, blending= 'additive', colormap='inferno' )
                    except:
                        pass   

        except:
             pass            

    def show_csv(self, imagename, csv_event_name, segimagedir = None, event_threshold = 0, use_dask = False, heatmapsteps = 0, nms_space = 0):
        
        csvname = None
        self.event_locations_size_dict.clear()
        self.size_locations = []
        self.score_locations = []
        self.event_locations = []
        self.confidence_locations = []
        for layer in list(self.viewer.layers):
                    if 'Detections'  in layer.name or layer.name in 'Detections' :
                            self.viewer.layers.remove(layer)   
        for (event_name,event_label) in self.key_categories.items():
                    if event_label > 0 and csv_event_name == event_name:
                            self.event_label = event_label     
                            csvname = self.csvdir + "/" + event_name + "Location" + (os.path.splitext(os.path.basename(imagename))[0] + '.csv')
        if csvname is not None:    
            
                self.event_name = csv_event_name                         
                self.dataset   = pd.read_csv(csvname, delimiter = ',')
                self.dataset_index =  self.dataset.index
                self.ax.cla()
                #Data is written as T, Y, X, Score, Size, Confidence
                T =  self.dataset[ self.dataset.keys()[0]][0:]
                Z =  self.dataset[ self.dataset.keys()[1]][0:]
                Y = self.dataset[self.dataset.keys()[2]][0:]
                X = self.dataset[self.dataset.keys()[3]][0:]
                Score = self.dataset[self.dataset.keys()[4]][0:]
                Size = self.dataset[self.dataset.keys()[5]][0:]
                Confidence = self.dataset[self.dataset.keys()[6]][0:]
                listtime = T.tolist()
                listz = Z.tolist()
                listy = Y.tolist()
                listx = X.tolist()
                listsize = Size.tolist()
                
                
                listscore = Score.tolist()
                listconfidence = Confidence.tolist()
                
            
                for i in (range(len(listtime))):
                        
                        tcenter = int(listtime[i])
                        zcenter = listz[i]
                        ycenter = listy[i]
                        xcenter = listx[i]
                        size = listsize[i]
                        score = listscore[i]
                        confidence = listconfidence[i]   
                        if score > event_threshold:
                                self.event_locations.append([int(tcenter),int(zcenter), int(ycenter), int(xcenter)])   

                                if int(tcenter) in self.event_locations_dict.keys():
                                    current_list = self.event_locations_dict[int(tcenter)]
                                    current_list.append([int(zcenter),int(ycenter), int(xcenter)])
                                    self.event_locations_dict[int(tcenter)] = current_list 
                                    self.event_locations_size_dict[(int(tcenter), int(zcenter), int(ycenter), int(xcenter))] = [size, score]
                                else:
                                    current_list = []
                                    current_list.append([int(zcenter),int(ycenter), int(xcenter)])
                                    self.event_locations_dict[int(tcenter)] = current_list    
                                    self.event_locations_size_dict[int(tcenter),int(zcenter), int(ycenter), int(xcenter)] = [size, score]

                                self.size_locations.append(size)
                                self.score_locations.append(score)
                                self.confidence_locations.append(confidence)
                point_properties = {'score' : np.array(self.score_locations), 'confidence' : np.array(self.confidence_locations),
                'size' : np.array(self.size_locations)}    
             
                name_remove = ('Detections','Location Map')
                for layer in list(self.viewer.layers):
                                    
                                    if  any(name in layer.name for name in name_remove):
                                            self.viewer.layers.remove(layer) 
                if len(self.score_locations) > 0:                             
                        self.viewer.add_points(self.event_locations,  properties = point_properties, symbol = 'square', blending = 'translucent_no_depth', name = 'Detections' + event_name, face_color = [0]*4, edge_color = "red") 
                        
               

                                       



def average_heat_map(image, sliding_window):

    j = 0
    for i in range(image.shape[0]):
        
              j = j + 1
              if i > 0:
                image[i,:] = np.add(image[i,:] , image[i - 1,:])
              if j == sliding_window:
                  image[i,:] = np.subtract(image[i,:] , image[i - 1,:])
                  j = 0
    return image          

                    
                    
           
                                
 
def TimedDistance(pointA, pointB):

    
     spacedistance = float(np.sqrt( (pointA[1] - pointB[1] ) * (pointA[1] - pointB[1] ) + (pointA[2] - pointB[2] ) * (pointA[2] - pointB[2] )  ))
     
     timedistance = float(np.abs(pointA[0] - pointB[0]))
     
     
     return spacedistance, timedistance
                
                
def GetMarkers(image):
    
    
    MarkerImage = np.zeros(image.shape)
    waterproperties = measure.regionprops(image)                
    Coordinates = [prop.centroid for prop in waterproperties]
    Coordinates = sorted(Coordinates , key=lambda k: [k[0], k[1]])
    coordinates_int = np.round(Coordinates).astype(int)
    MarkerImage[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))

    markers = morphology.dilation(MarkerImage, morphology.disk(2))        
   
    return markers  