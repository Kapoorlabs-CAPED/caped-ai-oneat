import os
import numpy as np
import csv
from scipy.spatial import KDTree

try:
    from pathlib import Path
    Path().expanduser()
except (ImportError,AttributeError):
    from pathlib2 import Path

try:
    import tempfile
    tempfile.TemporaryDirectory

except (ImportError,AttributeError):
    from backports import tempfile









def compute_nearest(box, boxes, secondboxes, mindist = 20):
    
    xbox = box[4]
    ybox = box[5]
    
    
    pts = []
    if boxes is not None:
     for i in range(0, boxes.shape[0]):
        xboxes = boxes[i,4]
        yboxes = boxes[i,5]
        center = (xboxes, yboxes)
        pts.append(center)
    secondpts = []
    if secondboxes is not None:
     for i in range(0, secondboxes.shape[0]):
        xboxes = secondboxes[i,4]
        yboxes = secondboxes[i,5]
        center = (xboxes, yboxes)
        secondpts.append(center)    
    indexA = None
    indexB = None    
    if len(pts)> 0:   
      pts = np.array(pts)
      Target = KDTree(pts)
      if Target is not None:     
        idx = Target.query_ball_point([xbox, ybox], r = mindist)
        if idx is not None :
         #Return the list of nearest points
         Nearest = pts[idx]
         SortNearest = np.sort(Nearest[::-1])
         if len(SortNearest)>1: 
          NearestXY = SortNearest[0]
          if boxes is not None:   
           for i in range(0, boxes.shape[0]):
             if(boxes[i,4] == NearestXY[0] and boxes[i,5] == NearestXY[1] ):
                indexA = i
      if len(secondpts)> 0:
       secondpts = np.array(secondpts)
       secondTarget = KDTree(secondpts)
       #Get id array of all nearest points

       if secondTarget is not None: 
        secondidx = secondTarget.query_ball_point([xbox, ybox], r = mindist)
 
       if secondidx is not None: 
        secondNearest = secondpts[secondidx]
        #Sort the list of nearest points
     
        SortsecondNearest = np.sort(secondNearest[::-1])

        if len(SortsecondNearest>1): 
         secondNearestXY = SortsecondNearest[0] 
       
         if secondboxes is not None:      
          for i in range(0, secondboxes.shape[0]):
           if(secondboxes[i,4] == secondNearestXY[0] and secondboxes[i,5] == secondNearestXY[1]):
              indexB = i

 
       
    return indexA, indexB  





def drawimage(eventlist, basedirResults, fname, Label):
   if eventlist is not None: 
    
      
    
     LocationXlist = []
     LocationYlist = []
     Timelist = []
    
     for i in range(0, len(eventlist)):
                    location,time,Name = eventlist[i]
                    (x,y) = location
                    returntime = int(time) 
                    
                    
                    LocationXlist.append(location[0])
                    LocationYlist.append(location[1])
                    Timelist.append(returntime)
                  
     Event_Count = np.column_stack([LocationXlist, LocationYlist,Timelist]) 
     Event_data = []
     
     
     for line in Event_Count:
        Event_data.append(line)
        writer = csv.writer(open(basedirResults + "/" + str(Label) + "LocationEventCounts" + (os.path.splitext(os.path.basename(fname))[0])  +".csv", "a"))
        writer.writerows(Event_data)
        Event_data = []
    


                 

                        



def StaticEvents(Image,  timelist, TimeFrames, boxes, originalsize, threshold, timepoint ,basedirResults, axes, fname, AppendName):
    
    originalsizeX, originalsizeY = originalsize
    
    
    EventList, EventBoxes = NMSSpace(boxes, originalsizeX, originalsizeY, threshold)
    

    
    return [], EventList
    
def DynamicEvents(Image,  timelist, TimeFrames, boxes, originalsize, threshold, timepoint, basedirResults, axes, fname, AppendName):
    
    originalsizeX, originalsizeY = originalsize
    EventList, boxes = NMSSpace(boxes, originalsizeX, originalsizeY, threshold)
    if len(EventList) > 0:
     boxes = boxes.tolist()
    EventList = []
    if timepoint%(TimeFrames + 2) == 0 and timepoint!=0:
        
      EventList, EventBoxes = NMSSpace(boxes, originalsizeX, originalsizeY, threshold)
      
      boxes = []
      
    
    return boxes, EventList  
    
def NMSSpace(boxes, originalsizeX, originalsizeY,threshold):
    
   if boxes is not None: 
    if len(boxes) == 0:
      return [], []
    else:
     boxes = np.array(boxes, dtype = float)
     assert boxes.shape[0] > 0
     if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)
     
     score = -boxes[:,6]
     idxs = np.argsort(score)

     pick = []
     while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
 
        
        distance = compute_dist(boxes[i], boxes[idxs[1:]])
      
        remove_idxs = np.where(distance < threshold)[0] + 1
        idxs = np.delete(idxs, remove_idxs)
        idxs = np.delete(idxs, 0)
        
     centerlist = []    

     for i in range(len(boxes[pick,0])):
 
       center = ( ((boxes[pick,4][i])) , ((boxes[pick,5][i])) )
       centerlist.append([center, boxes[pick,7][i],boxes[pick,8][i]] )
     if(boxes[pick,8][0] == 1):
       Name = 'Apoptosis'
     if(boxes[pick,8][0] == 2):
       Name = 'Division'
     if(boxes[pick,8][0] == 3):
       Name = 'MacroKitty'
     if(boxes[pick,8][0] == 4):
       Name = 'Non-Mature'
     if(boxes[pick,8][0] == 5):
       Name = 'Mature'
     print('Number of events:', len(centerlist),Name)

    
     return centerlist, boxes[pick]





def compute_dist(box, boxes):
    # Calculate intersection areas
    
    Xtarget = boxes[:, 4]
    Ytarget = boxes[:, 5]
    Ttarget = boxes[:, 7]
    
    Xsource = box[4]
    Ysource = box[5]
    Tsource = box[7]
    
    # If seperated in time the distance is made lower to avoid multi counting of events
    
        
    distance = (Xtarget - Xsource) * (Xtarget - Xsource) + (Ytarget - Ysource) * (Ytarget - Ysource) - (Ttarget - Tsource) * (Ttarget - Tsource)
    
    distance[ distance < 0] = 0
    
    return np.sqrt(distance)






