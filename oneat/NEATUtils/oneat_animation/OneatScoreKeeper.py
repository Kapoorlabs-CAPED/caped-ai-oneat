import pandas as pd
import numpy as np
from scipy import spatial
from skimage import measure
import os 
class ScoreModels:


     def __init__(self, segimage, predictions, groundtruth, thresholdscore = 1 -  1.0E-6,  thresholdspace = 10, thresholdtime = 2):

         self.segimage = segimage
         #A list of all the prediction csv files, path object
         self.predictions = predictions 
         #Approximate locations of the ground truth, Z co ordinate wil be ignored
         self.groundtruth = groundtruth
         self.thresholdscore = thresholdscore
         self.thresholdspace = thresholdspace 
         self.thresholdtime = thresholdtime
         self.Label_Coord = {}
         self.Coords = []
         self.CoordTree = None

     def model_scorer(self):

         Name = []
         TP = []
         FP = []
         FN = []
         self.LabelDict()
         columns = ['Model Name', 'True Positive', 'False Positive', 'False Negative']
         for csv_pred in self.predictions:
            self.csv_pred = csv_pred
            name = self.csv_pred.stem
            tp, fn, fp = self.TruePositives()
            
            Name.append(name)
            TP.append(tp)
            FN.append(fn)
            FP.append(fp)
         data = list(zip(Name, TP, FP, FN))

         df = pd.DataFrame(data, columns=columns)
         df.to_csv(self.csv_pred.parent + 'Model_Accuracy')
         return df

     def LabelDict(self):

         
         properties = measure.regionprops(self.segimage)                
         for prop in properties:
           self.Label_Coord[prop.label] = prop.centroid
           self.Coords.append(prop.centroid)
         self.CoordTree = spatial.cKDTree(self.Coords)


     def TruePositives(self):

            tp = 0
            dataset_pred  = pd.read_csv(self.csv_pred, delimiter = ',')
            T_pred = dataset_pred[dataset_pred.keys()[0]][0:]
            Y_pred = dataset_pred[dataset_pred.keys()[2]][0:]
            X_pred = dataset_pred[dataset_pred.keys()[3]][0:]
            Score_pred = dataset_pred[dataset_pred.keys()[4]][0:]
            
            listtime_pred = T_pred.tolist()
            listy_pred = Y_pred.tolist()
            listx_pred = X_pred.tolist()
            listscore_pred = Score_pred.tolist()
            location_pred = []
            for i in range(len(listtime_pred)):

                if listscore_pred[i] > self.thresholdscore:   
                    location_pred.append([listtime_pred[i], listy_pred[i], listx_pred[i]])

            tree = spatial.cKDTree(location_pred)


            dataset_gt  = pd.read_csv(self.groundtruth, delimiter = ',')
            T_gt = dataset_gt[dataset_gt.keys()[0]][0:]
            Z_gt = dataset_gt[dataset_gt.keys()[1]][0:]
            Y_gt = dataset_gt[dataset_gt.keys()[2]][0:]
            X_gt = dataset_gt[dataset_gt.keys()[3]][0:]

            listtime_gt = T_gt.tolist()
            listz_gt = Z_gt.tolist()
            listy_gt = Y_gt.tolist()
            listx_gt = X_gt.tolist()
            for i in range(len(listtime_gt)):
                
                index = (int(listtime_gt[i]), int(listz_gt[i]) , int(listy_gt[i]), int(listx_gt[i]))
                return_index = return_coordinates(self.segimage, index, self.Label_Coord, self.CoordTree)
                closestpoint = tree.query(return_index)
                spacedistance, timedistance = TimedDistance(return_index, location_pred[closestpoint[1]])
                
                if spacedistance < self.thresholdspace and timedistance < self.thresholdtime:
                    tp  = tp + 1
            
            fn = self.FalseNegatives(self.groundtruth)
            fp = self.FalsePositives(self.groundtruth)
            return tp/len(listtime_gt) * 100, fn, fp
        

     def FalseNegatives(self):
        
                

                        dataset_pred  = pd.read_csv(self.csv_pred, delimiter = ',')
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
                            
                            
                            if listscore_pred[i] > self.thresholdscore:
                              location_pred.append([listtime_pred[i], listy_pred[i], listx_pred[i]])

                        tree = spatial.cKDTree(location_pred)


                        dataset_gt  = pd.read_csv(self.groundtruth, delimiter = ',')
                        T_gt = dataset_gt[dataset_gt.keys()[0]][0:]
                        Z_gt = dataset_gt[dataset_gt.keys()[1]][0:]
                        Y_gt = dataset_gt[dataset_gt.keys()[2]][0:]
                        X_gt = dataset_gt[dataset_gt.keys()[3]][0:]

                        listtime_gt = T_gt.tolist()
                        listz_gt = Z_gt.tolist()
                        listy_gt = Y_gt.tolist()
                        listx_gt = X_gt.tolist()
                        fn = len(listtime_gt)
                        for i in range(len(listtime_gt)):
                            
                            index = (int(listtime_gt[i]), int(listz_gt[i])  ,int(listy_gt[i]), int(listx_gt[i]))
                            return_index = return_coordinates(self.segimage, index, self.Label_Coord, self.CoordTree)
                            closestpoint = tree.query(return_index)
                            spacedistance, timedistance = TimedDistance(return_index, location_pred[closestpoint[1]])

                            if spacedistance < self.thresholdspace and timedistance < self.thresholdtime:
                                fn  = fn - 1

                        return fn/len(listtime_gt) * 100
                    
                    
                    
     def FalsePositives(self, thresholdspace = 10, thresholdtime = 2):
        
                
            
                        dataset_pred  = pd.read_csv(self.csv_pred, delimiter = ',')
                        T_pred = dataset_pred[dataset_pred.keys()[0]][0:]
                        listtime_pred = T_pred.tolist()

                        dataset_gt  = pd.read_csv(self.groundtruth, delimiter = ',')
                        T_gt = dataset_gt[dataset_gt.keys()[0]][0:]
                        Z_gt = dataset_gt[dataset_gt.keys()[1]][0:]
                        Y_gt = dataset_gt[dataset_gt.keys()[2]][0:]
                        X_gt = dataset_gt[dataset_gt.keys()[3]][0:]

                        listtime_gt = T_gt.tolist()
                        listz_gt = Z_gt.tolist()
                        listy_gt = Y_gt.tolist()
                        listx_gt = X_gt.tolist()
                        location_gt = []
                        fp = len(listtime_pred)
                        
                        for i in range(len(listtime_gt)):
                        
                            location_gt.append([listtime_gt[i], listy_gt[i], listx_gt[i]])

                        tree = spatial.cKDTree(location_gt)
                        for i in range(len(listtime_pred)):
                            
                            index = (int(listtime_gt[i]), int(listz_gt[i])  ,int(listy_gt[i]), int(listx_gt[i]))
                            return_index = return_coordinates(self.segimage, index, self.Label_Coord, self.CoordTree)
                            closestpoint = tree.query(return_index)
                            spacedistance, timedistance = TimedDistance(return_index, location_gt[closestpoint[1]])

                            if spacedistance < thresholdspace and timedistance < thresholdtime:
                                fp  = fp - 1

                        return fp/len(listtime_pred) * 100
                    
                    
                                
 
def TimedDistance(pointA, pointB):

    
     spacedistance = float(np.sqrt( (pointA[1] - pointB[1] ) * (pointA[1] - pointB[1] ) + (pointA[2] - pointB[2] ) * (pointA[2] - pointB[2] )  ))
     
     timedistance = float(np.abs(pointA[0] - pointB[0]))
     
     
     return spacedistance, timedistance

def return_coordinates(image, coord, Label_Coord, CoordTree):

  
    print(coord, image.shape)
    label = image[CoordTree.query(coord)]
    print(label)
    return_coord = Label_Coord[label]

    return tuple(return_coord[0], return_coord[-2], return_coord[-1])      