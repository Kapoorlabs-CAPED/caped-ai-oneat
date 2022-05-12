import pandas as pd
import numpy as np
import os
import napari
from scipy import spatial
from skimage import measure


class OneatVisualization:

    def __init__(self, viewer: napari.Viewer, image_to_read, imagename, csv_event_name, plot_event_name,default_reader,key_categories, 
     savedir, savename, ax):

        self.viewer = viewer
        self.image_to_read = image_to_read
        self.imagename = imagename
        self.csv_event_name = csv_event_name
        self.plot_event_name = plot_event_name
        self.savedir = savedir
        self.savename = savename
        self.default_reader = default_reader
        self.key_categories = key_categories
        self.ax = ax
        
    def show_csv(self, segimagedir = None, event_threshold = 0, use_dask = False, heatmapsteps = 0):

        for (event_name,event_label) in self.key_categories.items():
                                
                                if event_label > 0 and self.csv_event_name == event_name:
                                     self.event_label = event_label                         
                                     for layer in list(self.viewer.layers):
                                          
                                         if 'Detections'  in layer.name or layer.name in 'Detections' :
                                                    self.viewer.layers.remove(layer)           
                                       
                                     
                                     csvname = self.savedir + "/" + event_name + "Location" + (os.path.splitext(os.path.basename(self.imagename))[0] + '.csv')
                                     
        dataset   = pd.read_csv(csvname, delimiter = ',')
        self.dataset_index = dataset.index
        self.ax.cla()
        #Data is written as T, Y, X, Score, Size, Confidence
        T = dataset[dataset.keys()[0]][0:]
        Z = dataset[dataset.keys()[1]][0:]
        Y = dataset[dataset.keys()[2]][0:]
        X = dataset[dataset.keys()[3]][0:]
        Score = dataset[dataset.keys()[4]][0:]
        Size = dataset[dataset.keys()[5]][0:]
        Confidence = dataset[dataset.keys()[6]][0:]
        
        
        
                

        listtime = T.tolist()
        listz = Z.tolist()
        listy = Y.tolist()
        listx = X.tolist()
        
        listsize = Size.tolist()
        listscore = Score.tolist()
        listconfidence = Confidence.tolist()
        event_locations = []
        event_locations_dict = {}
        size_locations = []
        score_locations = []
        confidence_locations = []
       
        for i in (range(len(listtime))):
                
                tcenter = int(listtime[i])
                zcenter = listz[i]
                ycenter = listy[i]
                xcenter = listx[i]
                size = listsize[i]
                score = listscore[i]
                confidence = listconfidence[i]   
                if score > event_threshold[self.event_label]:
                        event_locations.append([int(tcenter), int(ycenter), int(xcenter)])   

                        if int(tcenter) in event_locations_dict.keys():
                            current_list = event_locations_dict[int(tcenter)]
                            current_list.append([int(ycenter), int(xcenter)])
                            event_locations_dict[int(tcenter)] = current_list 
                        else:
                            current_list = []
                            current_list.append([int(ycenter), int(xcenter)])
                            event_locations_dict[int(tcenter)] = current_list    

                        size_locations.append(size)
                        score_locations.append(score)
                        confidence_locations.append(confidence)
        point_properties = {'score' : np.array(score_locations), 'confidence' : np.array(confidence_locations)}    
        text_properties = {
        'text': self.event_name +': {score:.5f}' + '\n' + 'Confidence' +  ': {confidence:.5f}',
        'anchor': 'upper_left',
        'translation': [-5, 0],
        'size': 12,
        'color': 'pink',
    }
        
        for layer in list(self.viewer.layers):
                            
                            if 'Detections'  in layer.name or layer.name in 'Detections' :
                                    self.viewer.layers.remove(layer) 
        if len(score_locations) > 0:                             
                self.viewer.add_points(event_locations, size = size_locations , properties = point_properties, text = text_properties,  name = 'Detections' + self.event_name, face_color = [0]*4, edge_color = "red") 
                
        if segimagedir is not None:
                for layer in list(self.viewer.layers):
                    if isinstance(layer, napari.layers.Labels):
                             seg_image = layer
                location_image, cell_count = LocationMap(event_locations_dict, seg_image, use_dask, heatmapsteps)     
                try:
                    self.viewer.add_image(location_image, name= 'Location Map' + self.imagename, blending= 'additive' )
                except:
                        pass


def LocationMap(event_locations_dict, seg_image, use_dask, heatmapsteps):
       cell_count = {} 
       location_image = np.zeros(seg_image.shape)
       for i in range(seg_image.shape[0]):
            if use_dask:
                    current_seg_image = seg_image[i,:].compute()
            else:
                    current_seg_image = seg_image[i,:]
            waterproperties = measure.regionprops(current_seg_image)
            indices = [prop.centroid for prop in waterproperties]
            cell_count[int(i)] = len(indices)        

            if int(i) in event_locations_dict.keys():
                currentindices = event_locations_dict[int(i)]
                    
                
                if len(indices) > 0:
                    tree = spatial.cKDTree(indices)
                    if len(currentindices) > 0:
                        for j in range(0, len(currentindices)):
                            index = currentindices[j] 
                            closest_marker_index = tree.query(index)
                            current_seg_label = current_seg_image[int(indices[closest_marker_index[1]][0]), int(
                            indices[closest_marker_index[1]][1])]
                            if current_seg_label > 0:
                                all_pixels = np.where(current_seg_image == current_seg_label)
                                all_pixels = np.asarray(all_pixels)
                                for k in range(all_pixels.shape[1]):
                                    location_image[i,all_pixels[0,k], all_pixels[1,k]] = 1
            if i > 0:
                location_image[i,:] = np.add(location_image[i -1,:],location_image[i,:])
                
       return location_image, cell_count