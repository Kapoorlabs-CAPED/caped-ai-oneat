import pandas as pd
import numpy as np
import os
import napari
from scipy import spatial
from skimage import measure
from dask.array.image import imread as daskread
from tifffile import imread,  imwrite
from skimage import morphology
import csv
import matplotlib.pyplot as plt
import cv2
class OneatVisualization:

    def __init__(self, viewer: napari.Viewer,key_categories, 
     savedir, savename, ax, figure):

        self.viewer = viewer
        self.savedir = savedir
        self.savename = savename
        self.key_categories = key_categories
        self.ax = ax
        self.figure = figure
        self.dataset = None
        self.event_name = None
        self.cell_count = None      
        self.event_locations = []
        self.event_locations_dict = {}
        self.event_locations_score_dict = {}
        self.event_locations_score_nested_dict = {}
        self.size_locations = []
        self.score_locations = []
        self.confidence_locations = []

    # To prevent early detectin of events
    def cluster_points(self, nms_space, nms_time):

     for k,v in self.event_locations_dict.keys():

         currenttime = k
         event_locations = v
         tree = spatial.cKDTree(event_locations)
         for i in range(nms_time//2):
                backtime = currenttime - i
                if backtime in self.event_locations_dict.keys():
                    back_event_locations = self.event_locations_dict[int(backtime)]
                    back_event_locations_score_dict = self.event_locations_score_nested_dict[int(backtime)]
                    
                    for location in back_event_locations:
                        backscore = back_event_locations_score_dict[location] 
                        distance, nearest_location = tree.query(location)

                        if distance <= nms_space:
                            
                                current_event_locations_score_dict = self.event_locations_score_nested_dict[int(currenttime)]
                                currentscore = current_event_locations_score_dict[nearest_location]

                                if currentscore > backscore:
                                    self.event_locations_score_nested_dict[int(backtime)].pop(location)
                                else:
                                    self.event_locations_score_nested_dict[int(currenttime)].pop(nearest_location)    
                forwardtime = currenttime + i
                if forwardtime in self.event_locations_dict.keys():
                    forward_event_locations = self.event_locations_dict[int(forwardtime)]
                    forward_event_locations_score_dict = self.event_locations_score_nested_dict[int(forwardtime)]
                    
                    for location in forward_event_locations:
                        forwardscore = forward_event_locations_score_dict[location] 
                        distance, nearest_location = tree.query(location)

                        if distance <= nms_space:
                            
                                current_event_locations_score_dict = self.event_locations_score_nested_dict[int(currenttime)]
                                currentscore = current_event_locations_score_dict[nearest_location]

                                if currentscore > forwardscore:
                                    self.event_locations_score_nested_dict[int(forwardtime)].pop(location)
                                else:
                                    self.event_locations_score_nested_dict[int(currenttime)].pop(nearest_location)                  

    def show_plot(self, imagename, plot_event_name, event_count_plot, event_norm_count_plot, cell_count_plot, 
      segimagedir = None, event_threshold = 0 ):

        print('event threshold in plot', event_threshold)
        timelist = []
        eventlist= []
        normeventlist = []
        celllist = []
        self.ax.cla()
        image = None
        
        if self.dataset is not None:                             
               
                for layer in list(self.viewer.layers):
                    if isinstance(layer, napari.layers.Image):
                            image = layer.data
                    if isinstance(layer, napari.layers.Labels):
                            seg_image = layer.data    


                if image is not None:            
                        for i in range(0, image.shape[0]):
                            
                            currentT   = np.round(self.dataset["T"]).astype('int')
                            currentZ = np.round(self.dataset["Z"]).astype('int')
                            currentScore = self.dataset["Score"]
                            currentConf = self.dataset["Confidence"]
                            condition = currentT == i
                            condition_indices = self.dataset_index[condition]
                            conditionScore = currentScore[condition_indices]
                            score_condition = conditionScore > event_threshold
                            countT = len(conditionScore[score_condition])
                            timelist.append(i)
                            eventlist.append(countT)
                            if segimagedir is not None and seg_image is not None:
                                all_cells = self.cell_count[i]
                                celllist.append(all_cells)
                                normeventlist.append(countT/all_cells)
                        if plot_event_name == event_count_plot:    
                                self.ax.plot(timelist, eventlist, '-r')
                                self.ax.set_title(self.event_name + "Events")
                                self.ax.set_xlabel("Time")
                                self.ax.set_ylabel("Counts")
                                self.figure.canvas.draw()
                                self.figure.canvas.flush_events()
                                plt.savefig(self.savedir  + self.event_name + event_count_plot + (os.path.splitext(os.path.basename(imagename))[0]  + '.png'), dpi = 300)

                        if plot_event_name == event_norm_count_plot and len(normeventlist) > 0:    
                                self.ax.plot(timelist, normeventlist, '-r')
                                self.ax.set_title(self.event_name + "Normalized Events")
                                self.ax.set_xlabel("Time")
                                self.ax.set_ylabel("Normalized Counts")
                                self.figure.canvas.draw()
                                self.figure.canvas.flush_events()
                                plt.savefig(self.savedir  + self.event_name + event_norm_count_plot + (os.path.splitext(os.path.basename(imagename))[0]  + '.png'), dpi = 300)

                        if plot_event_name == cell_count_plot and len(celllist) > 0:    
                                self.ax.plot(timelist, celllist, '-r')
                                self.ax.set_title("Total Cell counts")
                                self.ax.set_xlabel("Time")
                                self.ax.set_ylabel("Total Cell Counts")
                                self.figure.canvas.draw()
                                self.figure.canvas.flush_events()
                                plt.savefig(self.savedir  + cell_count_plot + (os.path.splitext(os.path.basename(imagename))[0]  + '.png'), dpi = 300)        





    def show_image(self, 
                image_toread, 
                imagename, 
                segimagedir = None, 
                heatmapimagedir = None, 
                heatname = '_Heat', 
                start_project_mid = 0, 
                end_project_mid = 0, 
                use_dask = False):

        name_remove = ('Image', 'SegImage')
        for layer in list(self.viewer.layers):
                                         if  any(name in layer.name for name in name_remove):
                                                    self.viewer.layers.remove(layer)
        try:                                            
            if use_dask:                                      
                image = daskread(image_toread)[0]
            else:
                image = imread(image_toread)    
            
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
                        seg_image = daskread(segimagedir + imagename + '.tif')[0]
                    else:
                        seg_image = imread(segimagedir + imagename + '.tif')    
                    if len(seg_image.shape) == 4:
                        seg_image =  MidSlices(seg_image, start_project_mid, end_project_mid, use_dask, axis = 1)
                    self.viewer.add_labels(seg_image.astype('uint16'), name = 'SegImage'+ imagename)
            if len(image.shape) == 4:
                image =  MidSlices(image, start_project_mid, end_project_mid, use_dask, axis = 1)
            
            self.viewer.add_image(image, name= 'Image' + imagename )
            if heatmapimagedir is not None:
                    try:
                      self.viewer.add_image(heat_image, name= 'Image' + imagename + heatname, blending= 'additive', colormap='inferno' )
                    except:
                        pass   

        except:
             pass            

    def show_csv(self, imagename, csv_event_name, segimagedir = None, event_threshold = 0, use_dask = False, heatmapsteps = 0, nms_space = 0, nms_time = 0):
        csvname = None
        
        for (event_name,event_label) in self.key_categories.items():
                                
                                if event_label > 0 and csv_event_name == event_name:
                                     self.event_label = event_label                         
                                     for layer in list(self.viewer.layers):
                                          
                                         if 'Detections'  in layer.name or layer.name in 'Detections' :
                                                    self.viewer.layers.remove(layer)           
                                       
                                     
                                     csvname = self.savedir + "/" + event_name + "Location" + (os.path.splitext(os.path.basename(imagename))[0] + '.csv')
        if csvname is not None:    
            try:  
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
                                self.event_locations.append([int(tcenter), int(ycenter), int(xcenter)])   

                                if int(tcenter) in self.event_locations_dict.keys():
                                    current_list = self.event_locations_dict[int(tcenter)]
                                    current_list.append([int(ycenter), int(xcenter)])
                                    self.event_locations_dict[int(tcenter)] = current_list 
                                    self.event_locations_score_dict[(int(ycenter), int(xcenter))] = score
                                    self.event_locations_score_nested_dict[int(tcenter)] = self.event_locations_score_dict[(int(ycenter), int(xcenter))]
                                else:
                                    current_list = []
                                    current_list.append([int(ycenter), int(xcenter)])
                                    self.event_locations_dict[int(tcenter)] = current_list    
                                    self.event_locations_score_dict[(int(ycenter), int(xcenter))] = score
                                    self.event_locations_score_nested_dict[int(tcenter)] = self.event_locations_score_dict[(int(ycenter), int(xcenter))]


                                self.size_locations.append(size)
                                self.score_locations.append(score)
                                self.confidence_locations.append(confidence)
                point_properties = {'score' : np.array(self.score_locations), 'confidence' : np.array(self.confidence_locations)}    
                text_properties = {
                'text': event_name +': {score:.5f}' + '\n' + 'Confidence' +  ': {confidence:.5f}',
                'anchor': 'upper_left',
                'translation': [-5, 0],
                'size': 12,
                'color': 'pink',
            }
                name_remove = ('Detections','Location Map')
                for layer in list(self.viewer.layers):
                                    
                                    if  any(name in layer.name for name in name_remove):
                                            self.viewer.layers.remove(layer) 
                if len(self.score_locations) > 0:                             
                        self.viewer.add_points(self.event_locations, size = self.size_locations , properties = point_properties, text = text_properties,  name = 'Detections' + event_name, face_color = [0]*4, edge_color = "red") 
                        
                if segimagedir is not None:
                        for layer in list(self.viewer.layers):
                            if isinstance(layer, napari.layers.Labels):
                                    seg_image = layer.data

                                    location_image, self.cell_count = LocationMap(self.event_locations_dict, seg_image, use_dask, heatmapsteps)     
                                    try:
                                        self.viewer.add_image(location_image, name= 'Location Map' + imagename, blending= 'additive' )
                                    except:
                                            pass

            except:

                pass                            


def LocationMap(event_locations_dict, seg_image, use_dask, heatmapsteps):
       cell_count = {} 
       location_image = np.zeros(seg_image.shape)
       j = 0
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
            
       location_image = average_heat_map(location_image, heatmapsteps)


       return location_image, cell_count


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

def MidSlices(Image, start_project_mid, end_project_mid, use_dask, axis = 1):
    
    if use_dask:
       SmallImage = Image.compute().take(indices = range(Image.shape[axis]//2 - start_project_mid, Image.shape[axis]//2 + end_project_mid), axis = axis)
    else:
       SmallImage = Image.take(indices = range(Image.shape[axis]//2 - start_project_mid, Image.shape[axis]//2 + end_project_mid), axis = axis)    
    MaxProject = np.amax(SmallImage, axis = axis)
        
    return MaxProject       


def TruePositives(csv_gt, csv_pred, thresholdscore = 1 -  1.0E-6,  thresholdspace = 10, thresholdtime = 2):
    
            
            try:
                
                    tp = 0
                  

                    dataset_pred  = pd.read_csv(csv_pred, delimiter = ',')
                    dataset_pred_index = dataset_pred.index

                    T_pred = dataset_pred[dataset_pred.keys()[0]][0:]
                    Y_pred = dataset_pred[dataset_pred.keys()[1]][0:]
                    X_pred = dataset_pred[dataset_pred.keys()[2]][0:]
                    Score_pred = dataset_pred[dataset_pred.keys()[3]][0:]
                    
                    
                    listtime_pred = T_pred.tolist()
                    listy_pred = Y_pred.tolist()
                    listx_pred = X_pred.tolist()
                    listscore_pred = Score_pred.tolist()
                    location_pred = []
                    for i in range(len(listtime_pred)):

                        if listscore_pred[i] > thresholdscore:   
                            location_pred.append([listtime_pred[i], listy_pred[i], listx_pred[i]])

                    tree = spatial.cKDTree(location_pred)


                    dataset_gt  = pd.read_csv(csv_gt, delimiter = ',')
                    dataset_gt_index = dataset_gt.index

                    T_gt = dataset_gt[dataset_gt.keys()[0]][0:]
                    Y_gt = dataset_gt[dataset_gt.keys()[1]][0:]
                    X_gt = dataset_gt[dataset_gt.keys()[2]][0:]

                    listtime_gt = T_gt.tolist()
                    listy_gt = Y_gt.tolist()
                    listx_gt = X_gt.tolist()
                    location_gt = []
                    for i in range(len(listtime_gt)):
                        
                        index = [float(listtime_gt[i]), float(listy_gt[i]), float(listx_gt[i])]
                        closestpoint = tree.query(index)
                        spacedistance, timedistance = TimedDistance(index, location_pred[closestpoint[1]])
                        
                        if spacedistance < thresholdspace and timedistance < thresholdtime:
                            tp  = tp + 1
                    
                    fn = FalseNegatives(csv_pred, csv_gt, thresholdscore = thresholdscore, thresholdspace = thresholdspace, thresholdtime = thresholdtime)
                    fp = FalsePositives(csv_pred, csv_gt, thresholdscore = thresholdscore, thresholdspace = thresholdspace, thresholdtime = thresholdtime)
                    return tp/len(listtime_gt) * 100, fn, fp
                
            except:
                 
                 return 'File not found'
                 pass

 
def DownsampleData(image, DownsampleFactor):
                    
                    if DownsampleFactor!=1:  
                  
                        print('Downsampling Image in XY by', DownsampleFactor)
                        scale_percent = int(100/DownsampleFactor) # percent of original size
                        width = int(image.shape[2] * scale_percent / 100)
                        height = int(image.shape[1] * scale_percent / 100)
                        dim = (width, height)
                        smallimage = np.zeros([image.shape[0],  height,width])
                        for i in range(0, image.shape[0]):
                              # resize image
                              smallimage[i,:] = cv2.resize(image[i,:].astype('float32'), dim)         

                        return smallimage
                    else:

                        return image
                
def PatchGenerator(image,resultsdir,csv_gt,number_patches, patch_shape, size_tminus,size_tplus,DownsampleFactor = 1 ):
    
    
                    image = DownsampleData(image, DownsampleFactor)
                    dataset_gt  = pd.read_csv(csv_gt, delimiter = ',')
            
                    dataset_gt = dataset_gt.sample(frac = 1)
                    dataset_gt_index = dataset_gt.index
                    T_gt = dataset_gt[dataset_gt.keys()[0]][0:]
                    Y_gt = dataset_gt[dataset_gt.keys()[1]][0:]/DownsampleFactor
                    X_gt = dataset_gt[dataset_gt.keys()[2]][0:]/DownsampleFactor

                    listtime_gt = T_gt.tolist()
                    
                    listy_gt = Y_gt.tolist()
                    listx_gt = X_gt.tolist()
                    location_gt = []
                    fn = len(listtime_gt)
                    count = 0
                    Data = []
                    for i in range(len(listtime_gt)):
                        if count >  2 * number_patches:
                            break
                        time = int(float(listtime_gt[i])) - 1
                        y = float(listy_gt[i])
                        x = float(listx_gt[i])
                        
                        if x > 0.25 * image.shape[2] and x < 0.75* image.shape[2] and y > 0.25 * image.shape[1] and y < 0.75* image.shape[1]:
                                crop_Xminus = x - int(patch_shape[0] / 2)
                                crop_Xplus = x + int(patch_shape[0] / 2)
                                crop_Yminus = y - int(patch_shape[1] / 2)
                                crop_Yplus = y + int(patch_shape[1] / 2)

                          
                                randomy = np.random.randint(min(0.25 * image.shape[2],0.25 * image.shape[1]), high=max(0.25 * image.shape[2],0.25 * image.shape[1]))
                                randomx = np.random.randint(min(0.25 * image.shape[2],0.25 * image.shape[1]), high=max(0.25 * image.shape[2],0.25 * image.shape[1]))
                                random_crop_Xminus = randomx - int(patch_shape[0] / 2)
                                random_crop_Xplus = randomx + int(patch_shape[0] / 2)
                                random_crop_Yminus = randomy - int(patch_shape[1] / 2)
                                random_crop_Yplus = randomy + int(patch_shape[1] / 2)

                                region = (slice(int(time - size_tminus),int(time + size_tplus  + 1)),slice(int(crop_Yminus), int(crop_Yplus)),
                                          slice(int(crop_Xminus), int(crop_Xplus)))

                                random_region = (slice(int(time - size_tminus),int(time + size_tplus  + 1)),slice(int(random_crop_Yminus), int(random_crop_Yplus)),
                                          slice(int(random_crop_Xminus), int(random_crop_Xplus)))


                                crop_image = image[region] 
                                random_crop_image = image[random_region]
                                if(crop_image.shape[0] == size_tplus + size_tminus + 1 and crop_image.shape[1]== patch_shape[1] and crop_image.shape[2]== patch_shape[0]):
                                      Data.append([time, y * DownsampleFactor, x * DownsampleFactor])
                                      imwrite(resultsdir + 'Skeletor' + 'T' +  str(time) + 'Y' + str(y*DownsampleFactor) + 'X' + str(x*DownsampleFactor) + '.tif', crop_image.astype('float16'),metadata={'axes': 'TYX'})                
                                count = count + 1        
                                if(random_crop_image.shape[0] == size_tplus + size_tminus + 1 and random_crop_image.shape[1]== patch_shape[1] and random_crop_image.shape[2]== patch_shape[0]):
                                      Data.append([time, randomy * DownsampleFactor, randomx * DownsampleFactor])
                                      imwrite(resultsdir + 'Skeletor' + 'T' + str(time) + 'Y' + str(randomy*DownsampleFactor) + 'X' + str(randomx*DownsampleFactor) + '.tif', random_crop_image.astype('float16'),metadata={'axes': 'TYX'})
                                count = count + 1 
                    
                    writer = csv.writer(open(resultsdir + '/' + ('GTLocator') + ".csv", "w"))
                    writer.writerows(Data)
                    
                    
def FalseNegatives(csv_pred, csv_gt, thresholdscore = 1 -  1.0E-6, thresholdspace = 10, thresholdtime = 2):
    
            
            try:
                
                    
                  

                    dataset_pred  = pd.read_csv(csv_pred, delimiter = ',')
                    dataset_pred_index = dataset_pred.index

                    T_pred = dataset_pred[dataset_pred.keys()[0]][0:]
                    Y_pred = dataset_pred[dataset_pred.keys()[1]][0:]
                    X_pred = dataset_pred[dataset_pred.keys()[2]][0:]
                    Score_pred = dataset_pred[dataset_pred.keys()[3]][0:]
                    
                    listtime_pred = T_pred.tolist()
                    listy_pred = Y_pred.tolist()
                    listx_pred = X_pred.tolist()
                    listscore_pred = Score_pred.tolist()
                    location_pred = []
                    for i in range(len(listtime_pred)):
                        
                        
                        if listscore_pred[i] > thresholdscore:
                           location_pred.append([listtime_pred[i], listy_pred[i], listx_pred[i]])

                    tree = spatial.cKDTree(location_pred)


                    dataset_gt  = pd.read_csv(csv_gt, delimiter = ',')
                    dataset_gt_index = dataset_gt.index

                    T_gt = dataset_gt[dataset_gt.keys()[0]][0:]
                    Y_gt = dataset_gt[dataset_gt.keys()[1]][0:]
                    X_gt = dataset_gt[dataset_gt.keys()[2]][0:]

                    listtime_gt = T_gt.tolist()
                    listy_gt = Y_gt.tolist()
                    listx_gt = X_gt.tolist()
                    location_gt = []
                    fn = len(listtime_gt)
                    for i in range(len(listtime_gt)):
                        
                        index = [float(listtime_gt[i]), float(listy_gt[i]), float(listx_gt[i])]
                        closestpoint = tree.query(index)
                        spacedistance, timedistance = TimedDistance(index, location_pred[closestpoint[1]])

                        if spacedistance < thresholdspace and timedistance < thresholdtime:
                            fn  = fn - 1

                            


                    return fn/len(listtime_gt) * 100
                
            except:
                 
                 return 'File not found'
                 pass             
                
def FalsePositives(csv_pred, csv_gt, thresholdscore = 1 -  1.0E-6, thresholdspace = 10, thresholdtime = 2):
    
            
            try:
                
                    
                  

                    dataset_pred  = pd.read_csv(csv_pred, delimiter = ',')
                    dataset_pred_index = dataset_pred.index

                    T_pred = dataset_pred[dataset_pred.keys()[0]][0:]
                    Y_pred = dataset_pred[dataset_pred.keys()[1]][0:]
                    X_pred = dataset_pred[dataset_pred.keys()[2]][0:]
                    Score_pred = dataset_pred[dataset_pred.keys()[3]][0:]
                    
                    listtime_pred = T_pred.tolist()
                    listy_pred = Y_pred.tolist()
                    listx_pred = X_pred.tolist()
                    listscore_pred = Score_pred.tolist()
                    location_pred = []
                    


                    dataset_gt  = pd.read_csv(csv_gt, delimiter = ',')
                    dataset_gt_index = dataset_gt.index

                    T_gt = dataset_gt[dataset_gt.keys()[0]][0:]
                    Y_gt = dataset_gt[dataset_gt.keys()[1]][0:]
                    X_gt = dataset_gt[dataset_gt.keys()[2]][0:]

                    listtime_gt = T_gt.tolist()
                    listy_gt = Y_gt.tolist()
                    listx_gt = X_gt.tolist()
                    location_gt = []
                    fp = len(listtime_pred)
                    
                    for i in range(len(listtime_gt)):
                        
                     
                           location_gt.append([listtime_gt[i], listy_gt[i], listx_gt[i]])

                    tree = spatial.cKDTree(location_gt)
                    for i in range(len(listtime_pred)):
                        
                        index = [float(listtime_pred[i]), float(listy_pred[i]), float(listx_pred[i])]
                        closestpoint = tree.query(index)
                        spacedistance, timedistance = TimedDistance(index, location_gt[closestpoint[1]])

                        if spacedistance < thresholdspace and timedistance < thresholdtime:
                            fp  = fp - 1

                            


                    return fp/len(listtime_pred) * 100
                
            except:
                 
                 return 'File not found'
                 pass             
                                
 
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






  