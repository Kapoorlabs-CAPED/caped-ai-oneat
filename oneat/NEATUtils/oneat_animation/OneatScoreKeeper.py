import pandas as pd
import numpy as np
from scipy import spatial
from skimage import measure
import os 
class ScoreModels:


     def __init__(self, predictions, groundtruth, thresholdscore = 1 -  1.0E-6,  thresholdspace = 10, thresholdtime = 2):

         #A list of all the prediction csv files, path object
         self.predictions = predictions 
         #Approximate locations of the ground truth, Z co ordinate wil be ignored
         self.groundtruth = groundtruth
         self.thresholdscore = thresholdscore
         self.thresholdspace = thresholdspace 
         self.thresholdtime = thresholdtime
         self.location_pred = []
         self.location_gt = []

         self.listtime_pred = []
         self.listy_pred = []
         self.listx_pred = []
         self.listscore_pred = []

         self.listtime_gt = []
         self.listy_gt = []
         self.listx_gt = []


     def model_scorer(self):

         Name = []
         TP = []
         FP = []
         FN = []
         GT = []
         Score_thresh= []
         Pred = []
         columns = ['Model Name', 'True Positive', 'False Positive', 'False Negative', 'Total Predictions', 'GT predictions', 'Score Threshold']
         

         dataset_gt  = pd.read_csv(self.groundtruth, delimiter = ',')
         T_gt = dataset_gt[dataset_gt.keys()[0]][0:]
         Y_gt = dataset_gt[dataset_gt.keys()[2]][0:]
         X_gt = dataset_gt[dataset_gt.keys()[3]][0:]

         self.listtime_gt = T_gt.tolist()
         self.listy_gt = Y_gt.tolist()
         self.listx_gt = X_gt.tolist()
         for i in range(len(self.listtime_gt)):

              self.location_gt.append([self.listtime_gt[i], self.listy_gt[i], self.listx_gt[i]])
         

         for csv_pred in self.predictions:
            self.location_pred = []

            self.listtime_pred = []
            self.listy_pred = []
            self.listx_pred = []
            self.listscore_pred = []

            self.csv_pred = csv_pred
            name = self.csv_pred.stem
            dataset_pred  = pd.read_csv(self.csv_pred, delimiter = ',')
            T_pred = dataset_pred[dataset_pred.keys()[0]][0:]
            Y_pred = dataset_pred[dataset_pred.keys()[2]][0:]
            X_pred = dataset_pred[dataset_pred.keys()[3]][0:]
            Score_pred = dataset_pred[dataset_pred.keys()[4]][0:]
        
            self.listtime_pred = T_pred.tolist()
            self.listy_pred = Y_pred.tolist()
            self.listx_pred = X_pred.tolist()
            self.listscore_pred = Score_pred.tolist()

            for i in range(len(self.listtime_pred)):

                if float(self.listscore_pred[i]) >= float(self.thresholdscore):   
                    self.location_pred.append([int(self.listtime_pred[i]), int(self.listy_pred[i]), int(self.listx_pred[i])])
            tp, fn, fp, pred, gt = self.TruePositives()
            
            Name.append(name)
            TP.append(tp)
            FN.append(fn)
            FP.append(fp)
            GT.append(gt)
            Pred.append(pred)
            Score_thresh.append(self.thresholdscore)
         data = list(zip(Name, TP, FP, FN, Pred, GT, Score_thresh))

         df = pd.DataFrame(data, columns=columns)
         df.to_csv(str(self.csv_pred.parent) + '_Model_Accuracy' + '.csv')
         return df

     

     def TruePositives(self):

            tp = 0
            fp = 0
            tree = spatial.cKDTree(self.location_gt)
            for i in range(len(self.location_pred)):
                
                return_index = self.location_pred[i]
                closestpoint = tree.query(return_index)
                spacedistance, timedistance = TimedDistance(return_index, self.location_gt[closestpoint[1]])
                    
                if spacedistance < self.thresholdspace and timedistance < self.thresholdtime:
                        tp  = tp + 1
                else:
                        fp = fp + 1        
            
            fn = self.FalseNegatives()
            return tp, fn, fp, len(self.location_pred), len(self.location_gt)
        

     def FalseNegatives(self):
        
                        tree = spatial.cKDTree(self.location_pred)
                        fn = 0
                        for i in range(len(self.listtime_gt)):
                            
                            return_index = (int(self.listtime_gt[i]),int(self.listy_gt[i]), int(self.listx_gt[i]))
                            closestpoint = tree.query(return_index)
                            spacedistance, timedistance = TimedDistance(return_index, self.location_pred[closestpoint[1]])

                            if spacedistance > self.thresholdspace or timedistance > self.thresholdtime:
                                    fn  = fn + 1

                        return fn
                    
                    
    
                    
                    
                                
 
def TimedDistance(pointA, pointB):

    
     spacedistance = float(np.sqrt( (pointA[1] - pointB[1] ) * (pointA[1] - pointB[1] ) + (pointA[2] - pointB[2] ) * (pointA[2] - pointB[2] )  ))
     
     timedistance = float(np.abs(pointA[0] - pointB[0]))
     
     
     return spacedistance, timedistance

   