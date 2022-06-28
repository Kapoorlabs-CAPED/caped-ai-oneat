import sys
sys.path.append("../NEAT")
import csv
import numpy as np
from tifffile import imread, imwrite 
import pandas as pd
import os
from tqdm import tqdm
import glob
import json
from skimage.measure import regionprops
from skimage import measure
from pathlib import Path
from sklearn.model_selection import train_test_split
from .helpers import  normalizeFloatZeroOne
from PIL import Image    
"""
@author: Varun Kapoor
In this program we create training movies and training images for ONEAT. The training data comprises of images and text labels attached to them.
TrainingMovies: This program is for action recognition training data creation. The inputs are the training image, the corresponding integer labelled segmentation image,
csv file containing time, ylocation, xlocation, angle (optional)
Additional parameters to be supplied are the 
1) sizeTminus: action events are centered at the time location, this parameter is the start time of the time volume the network carved out from the image.
2) sizeTplus: this parameter is the end of the time volume to be carved out from the image.
3) total_categories: It is the number of total action categories the network is supposed to predict, Vanilla ONEAT has these labels:
   0: NormalEvent
   1: ApoptosisEvent
   2: DivisionEvent
   3: Macrocheate as static dynamic event
   4: Non MatureP1 cells as static dynamic event
   5: MatureP1 cells as static dynamic event
    
TrainingImages: This program is for cell type recognition training data creation. The inputs are the trainng image, the corresponding integer labelled segmentation image,
Total categories for cell classification part of vanilla ONEAT are:
    0: Normal cells
    1: Central time frame of apoptotic cell
    2: Central time frame of dividing cell
    3: Macrocheates
    4: Non MatureP1 cells
    5: MatureP1 cells
csv file containing time, ylocation, xlocation of that event/cell type
"""    
   
def SegFreeMovieLabelDataSet(image_dir, csv_dir, save_dir, static_name, static_label, csv_name_diff, crop_size,gridx = 1, gridy = 1, normPatch = False, yolo_v0 = False, yolo_v1 = True, yolo_v2 = False, tshift = 0, normalizeimage = True):
    
    
            raw_path = os.path.join(image_dir, '*tif')
            Csv_path = os.path.join(csv_dir, '*csv')
            files_raw = glob.glob(raw_path)
            files_raw.sort
            filesCsv = glob.glob(Csv_path)
            filesCsv.sort
            Path(save_dir).mkdir(exist_ok=True)
            total_categories = len(static_name)
                
            
            
            for fname in files_raw:
                          
                         name = os.path.basename(os.path.splitext(fname)[0])  
                         
                        
                         for csvfname in filesCsv:
                                 count = 0  
                                 Csvname =  os.path.basename(os.path.splitext(csvfname)[0])
                                 
                                 for i in  range(0, len(static_name)):
                                     event_name = static_name[i]
                                     trainlabel = static_label[i]
                                     classfound = (Csvname == csv_name_diff +  event_name+ name )  
                                     if classfound:
                                                    print(Csvname)
                                                    image = imread(fname)
                                                    if normPatch ==False and normalizeimage == True:
                                                       image = normalizeFloatZeroOne( image.astype('float32'),1,99.8)
                                                    dataset = pd.read_csv(csvfname)
                                                    time = dataset[dataset.keys()[0]][1:]
                                                    y = dataset[dataset.keys()[1]][1:]
                                                    x = dataset[dataset.keys()[2]][1:]
                                                                                
                                                                             
                                                    #Categories + XYHW + Confidence 
                                                    for (key, t) in time.items():
                                                       try: 
                                                          SimpleMovieMaker(t, y[key], x[key], image, crop_size,gridx, gridy, total_categories, trainlabel, name+ event_name + str(count), save_dir, normPatch,yolo_v0, yolo_v1, yolo_v2, tshift) 
                                                          count = count + 1
                                                        
                                                       except:
                                                        
                                                           pass
                                                        
def SegFreeMovieLabelDataSet4D(image_dir, csv_dir, save_dir, static_name, static_label, csv_name_diff, crop_size, gridx = 1, gridy = 1, normPatch = False, yolo_v0 = False, yolo_v1 = True, yolo_v2 = False, tshift = 0, normalizeimage = True):
    
    
            raw_path = os.path.join(image_dir, '*tif')
            Csv_path = os.path.join(csv_dir, '*csv')
            files_raw = glob.glob(raw_path)
            files_raw.sort
            filesCsv = glob.glob(Csv_path)
            filesCsv.sort
            Path(save_dir).mkdir(exist_ok=True)
            total_categories = len(static_name)
                
            
            
            for fname in files_raw:
                          
                         name = os.path.basename(os.path.splitext(fname)[0])  
                         
                        
                         for csvfname in filesCsv:
                                 count = 0  
                                 Csvname =  os.path.basename(os.path.splitext(csvfname)[0])
                                 
                                 for i in  range(0, len(static_name)):
                                     event_name = static_name[i]
                                     trainlabel = static_label[i]
                                     classfound = (Csvname == csv_name_diff +  event_name+ name )  
                                     if classfound:
                                                    print(Csvname)
                                                    image = imread(fname)
                                                    dataset = pd.read_csv(csvfname)
                                                    time = dataset[dataset.keys()[0]][1:]
                                                    z = dataset[dataset.keys()[1]][1:]
                                                    y = dataset[dataset.keys()[2]][1:]
                                                    x = dataset[dataset.keys()[3]][1:]
                                                                                
                                                                             
                                                    #Categories + XYHW + Confidence 
                                                    for (key, t) in time.items():
                                                       try: 
                                                          SimpleMovieMaker4D(normalizeimage, t, z[key], y[key], x[key], image, crop_size,gridx, gridy, total_categories, trainlabel, name+ event_name + str(count), save_dir, normPatch,yolo_v0, yolo_v1, yolo_v2, tshift) 
                                                          count = count + 1
                                                        
                                                       except:
                                                        
                                                           pass
def SimpleMovieMaker(time, y, x, image, crop_size,gridx, gridy, total_categories, trainlabel, name, save_dir, normPatch,yolo_v0, yolo_v1, yolo_v2, tshift):
    
       sizex, sizey, size_tminus, size_tplus = crop_size
       
       imagesizex = sizex * gridx
       imagesizey = sizey * gridy
       
       shiftNone = [0,0]
       AllShifts = [shiftNone]


       time = time - tshift
       if time > 0:
               for shift in AllShifts:

                        newname = name + 'shift' + str(shift)
                        Event_data = []
                        
                        if yolo_v0:
                            Label = np.zeros([total_categories + 5])
                        if yolo_v1:    
                            Label = np.zeros([total_categories + 6])
                        if yolo_v2:
                            Label = np.zeros([total_categories + 7])
                        Label[trainlabel] = 1
                        
                        newcenter = (y - shift[1],x - shift[0] )
                        if x + shift[0]> sizex/2 and y + shift[1] > sizey/2 and x + shift[0] + int(imagesizex/2) < image.shape[2] and y + shift[1]+ int(imagesizey/2) < image.shape[1] and time > size_tminus and time + size_tplus + 1 < image.shape[0]:
                                crop_xminus = x  - int(imagesizex/2)
                                crop_xplus = x  + int(imagesizex/2)
                                crop_yminus = y  - int(imagesizey/2)
                                crop_yplus = y   + int(imagesizey/2)
                                # Cut off the region for training movie creation
                                region =(slice(int(time - size_tminus),int(time + size_tplus  + 1)),slice(int(crop_yminus)+ shift[1], int(crop_yplus)+ shift[1]),
                                      slice(int(crop_xminus) + shift[0], int(crop_xplus) + shift[0]))
                                #Define the movie region volume that was cut
                                crop_image = image[region]   
                                if normPatch:
                                    crop_image = normalizeFloatZeroOne( crop_image.astype('float32'),1,99.8)
                                seglocationx = (newcenter[1] - crop_xminus)
                                seglocationy = (newcenter[0] - crop_yminus)
                                Label[total_categories] =  seglocationx/sizex
                                Label[total_categories + 1] = seglocationy/sizey
                                Label[total_categories + 2] = (size_tminus) / (size_tminus + size_tplus)             
                                Label[total_categories + 3] = 1
                                Label[total_categories + 4] = 1
                                Label[total_categories + 5] = 1
       

                                #Write the image as 32 bit tif file 
                                if(crop_image.shape[0] == size_tplus + size_tminus + 1 and crop_image.shape[1]== imagesizey and crop_image.shape[2]== imagesizex):

                                           imwrite((save_dir + '/' + newname + '.tif'  ) , crop_image.astype('float32'))    
                                           Event_data.append([Label[i] for i in range(0,len(Label))])
                                           if(os.path.exists(save_dir + '/' + (newname) + ".csv")):
                                                        os.remove(save_dir + '/' + (newname) + ".csv")
                                           writer = csv.writer(open(save_dir + '/' + (newname) + ".csv", "a"))
                                           writer.writerows(Event_data)

def SimpleMovieMaker4D(normalizeimage, time, z, y, x, image, crop_size, gridx, gridy, total_categories, trainlabel, name, save_dir, normPatch,yolo_v0, yolo_v1, yolo_v2, tshift):
    
       sizex, sizey, size_tminus, size_tplus = crop_size
       
       imagesizex = sizex * gridx
       imagesizey = sizey * gridy
       
       shiftNone = [0,0]
       AllShifts = [shiftNone]

       
       time = time - tshift

       image = image[:,z,:,:]
       if normalizeimage:
                    image = normalizeFloatZeroOne( image.astype('float32'),1,99.8)

       if time > 0:
               
               for shift in AllShifts:

                        newname = name + 'shift' + str(shift)
                        Event_data = []
                        
                        if yolo_v0:
                            Label = np.zeros([total_categories + 5])
                        if yolo_v1:    
                            Label = np.zeros([total_categories + 6])
                        if yolo_v2:
                            Label = np.zeros([total_categories + 7])
                        Label[trainlabel] = 1
                        
                        newcenter = (y - shift[1],x - shift[0] )
                        if x + shift[0]> sizex/2 and y + shift[1] > sizey/2 and x + shift[0] + int(imagesizex/2) < image.shape[2] and y + shift[1]+ int(imagesizey/2) < image.shape[1] and time > size_tminus and time + size_tplus + 1 < image.shape[0]:
                                crop_xminus = x  - int(imagesizex/2)
                                crop_xplus = x  + int(imagesizex/2)
                                crop_yminus = y  - int(imagesizey/2)
                                crop_yplus = y   + int(imagesizey/2)
                                # Cut off the region for training movie creation
                                region =(slice(int(time - size_tminus),int(time + size_tplus  + 1)),slice(int(crop_yminus)+ shift[1], int(crop_yplus)+ shift[1]),
                                      slice(int(crop_xminus) + shift[0], int(crop_xplus) + shift[0]))
                                #Define the movie region volume that was cut
                                crop_image = image[region]   
                                if normPatch:
                                    crop_image = normalizeFloatZeroOne( crop_image.astype('float32'),1,99.8)

                                seglocationx = (newcenter[1] - crop_xminus)
                                seglocationy = (newcenter[0] - crop_yminus)
                                Label[total_categories] =  seglocationx/sizex
                                Label[total_categories + 1] = seglocationy/sizey
                                Label[total_categories + 2] = (size_tminus) / (size_tminus + size_tplus)      
                                Label[total_categories + 3] = 1
                                Label[total_categories + 4] = 1
                                Label[total_categories + 5] = 1
       

                                #Write the image as 32 bit tif file 
                                if(crop_image.shape[0] == size_tplus + size_tminus + 1 and crop_image.shape[1]== imagesizey and crop_image.shape[2]== imagesizex):

                                           imwrite((save_dir + '/' + newname + '.tif'  ) , crop_image.astype('float32'))    
                                           Event_data.append([Label[i] for i in range(0,len(Label))])
                                           if(os.path.exists(save_dir + '/' + (newname) + ".csv")):
                                                        os.remove(save_dir + '/' + (newname) + ".csv")
                                           writer = csv.writer(open(save_dir + '/' + (newname) + ".csv", "a"))
                                           writer.writerows(Event_data)                                                                    
    
def loadResizeImgs(im, size):
    
          
            w,h = im.size
            if w < h:
                im = im.crop((0, (h-w)/2, w, (h+w)/2))
            elif w > h:
                im = im.crop(((w-h+1)/2,0, (w+h)/2, h))

            return np.array(im.resize(size, Image.BILINEAR))
            

def Folder_to_oneat(dir, trainlabel, trainname, total_categories, size, save_dir):

        Label = np.zeros([total_categories]) 
        
        count = 0
        
        files = sorted(glob.glob(dir + "/" + '*.png'))
        for i in tqdm(range(len(files))):
                file = files[i]
                try:
                    Event_data = [] 
                   
                    img = Image.open(file)
                    img = loadResizeImgs(img, size)
                    Name = str(trainname) + os.path.basename(os.path.splitext(file)[0])
                    image = normalizeFloatZeroOne( img.astype('float32'),1,99.8)
                    Label[trainlabel] = 1
                    imwrite((save_dir + '/' + Name + str(count)  + '.tif'  ) , image.astype('float32'))  
                    Event_data.append([Label[i] for i in range(0,len(Label))])
                    if(os.path.exists(save_dir + '/' + Name + str(count) + ".csv")):
                        os.remove(save_dir + '/' + Name + str(count) + ".csv")
                    writer = csv.writer(open(save_dir + '/' + Name + str(count) + ".csv", "a"))
                    writer.writerows(Event_data)
                    count = count + 1
                except Exception as e:
                    print("[WW] ", str(e))
                    continue
           
            
            

def Midog_to_oneat(midog_folder, annotation_file,event_type_name_label, all_ids, crop_size, save_dir):

    rows = []
    annotations = {}
    id_to_tumortype = {id:list(k for k in all_ids if id in all_ids[k])[0] for id in range(1,406)}
    with open(annotation_file) as f:
        data = json.load(f)
    
        #categories = {cat["id"]: cat["name"] for cat in data["categories"]}
        categories = {1: 'mitotic figure', 2: 'hard negative'}
        total_categories = len(event_type_name_label.keys())
        for row in data["images"]:
            file_name = row["file_name"]
            image_id = row["id"]
            width = row["width"]
            height = row["height"]
            tumortype = id_to_tumortype[image_id]
            
            for annotation in [anno for anno in data['annotations'] if anno["image_id"] == image_id]:
                box = annotation["bbox"]
                cat = annotation["category_id"]

                rows.append([image_id, width, height, box, cat, tumortype])
            annotations[file_name] = rows
    


    
    count = 0 
    for tumortype, ids in zip(list(all_ids.keys()), list(all_ids.values())):

        for image_id in ids:

            
            
            file_path = midog_folder + "/" +  f"{image_id:03d}.tiff"
            Name = os.path.basename(os.path.splitext(file_path)[0])
           
            img = imread(file_path)
            image = normalizeFloatZeroOne( img.astype('float32'),1,99.8)
            image_annotation_array = annotations[Name + '.tiff']
            image_id += 1

    for image_annotation in image_annotation_array:
        
        Label = np.zeros([total_categories + 5]) 
        Event_data = []
        image_id, image_width, image_height, box, cat, tumortype = image_annotation
        Name = str(image_id)
        x0, y0, x1, y1 = box
        height = y1 - y0
        width = x1 - x0
        x = (x0 + x1) //2
        y = (y0 + y1) //2
        # if cat == 1 then it is mitosis if cat == 2 it is hard negative
        if cat == 2:
            trainlabel = event_type_name_label[tumortype] + total_categories//2  
        if cat == 1 :
            trainlabel = event_type_name_label[tumortype]
        ImagesizeX, ImagesizeY = crop_size
        crop_Xminus = x  - int(ImagesizeX/2)
        crop_Xplus = x   + int(ImagesizeX/2)
        crop_Yminus = y  - int(ImagesizeY/2)
        crop_Yplus = y   + int(ImagesizeY/2)
        region =(slice(int(crop_Yminus), int(crop_Yplus)),
                                                    slice(int(crop_Xminus), int(crop_Xplus)))

        crop_image = image[region]      

        
        
        Label[trainlabel] = 1
        Label[total_categories] =  0.5
        Label[total_categories + 1] = 0.5
        Label[total_categories + 2] = width/ImagesizeX
        Label[total_categories + 3] = height/ImagesizeY
        Label[total_categories + 4] = 1 
        
        count = count + 1
        if(crop_image.shape[0]== ImagesizeY and crop_image.shape[1]== ImagesizeX):
                    imwrite((save_dir + '/' + Name + str(count)  + '.tif'  ) , crop_image.astype('float32'))  
                    Event_data.append([Label[i] for i in range(0,len(Label))])
                    if(os.path.exists(save_dir + '/' + Name + str(count) + ".csv")):
                        os.remove(save_dir + '/' + Name + str(count) + ".csv")
                    writer = csv.writer(open(save_dir + '/' + Name + str(count) + ".csv", "a"))
                    writer.writerows(Event_data)

def Midog_to_oneat_simple(midog_folder, annotation_file,event_type_name_label, all_ids, crop_size, save_dir):

    rows = []
    annotations = {}
    id_to_tumortype = {id:list(k for k in all_ids if id in all_ids[k])[0] for id in range(1,406)}
    with open(annotation_file) as f:
        data = json.load(f)
    
        #categories = {cat["id"]: cat["name"] for cat in data["categories"]}
        categories = {1: 'mitotic figure', 2: 'hard negative'}
        total_categories = len(event_type_name_label.keys())
        for row in data["images"]:
            file_name = row["file_name"]
            image_id = row["id"]
            width = row["width"]
            height = row["height"]
            tumortype = id_to_tumortype[image_id]
            
            for annotation in [anno for anno in data['annotations'] if anno["image_id"] == image_id]:
                box = annotation["bbox"]
                cat = annotation["category_id"]

                rows.append([image_id, width, height, box, cat, tumortype])
            annotations[file_name] = rows
    


    
    count = 0 
    for tumortype, ids in zip(list(all_ids.keys()), list(all_ids.values())):

        for image_id in ids:

            
            
            file_path = midog_folder + "/" +  f"{image_id:03d}.tiff"
            Name = os.path.basename(os.path.splitext(file_path)[0])
           
            img = imread(file_path)
            image = normalizeFloatZeroOne( img.astype('float32'),1,99.8)
            image_annotation_array = annotations[Name + '.tiff']
            image_id += 1

    for image_annotation in image_annotation_array:
        
        Label = np.zeros([total_categories]) 
        Event_data = []
        image_id, image_width, image_height, box, cat, tumortype = image_annotation
        Name = str(image_id)
        x0, y0, x1, y1 = box
        height = y1 - y0
        width = x1 - x0
        x = (x0 + x1) //2
        y = (y0 + y1) //2
        # if cat == 1 then it is mitosis if cat == 2 it is hard negative
        if cat == 2:
            trainlabel = event_type_name_label[tumortype] + total_categories//2  
        if cat == 1 :
            trainlabel = event_type_name_label[tumortype]
        ImagesizeX, ImagesizeY = crop_size
        crop_Xminus = x  - int(ImagesizeX/2)
        crop_Xplus = x   + int(ImagesizeX/2)
        crop_Yminus = y  - int(ImagesizeY/2)
        crop_Yplus = y   + int(ImagesizeY/2)
        region =(slice(int(crop_Yminus), int(crop_Yplus)),
                                                    slice(int(crop_Xminus), int(crop_Xplus)))

        crop_image = image[region]      

        
        
        Label[trainlabel] = 1
        
        
        count = count + 1
        if(crop_image.shape[0]== ImagesizeY and crop_image.shape[1]== ImagesizeX):
                    imwrite((save_dir + '/' + Name + str(count)  + '.tif'  ) , crop_image.astype('float32'))  
                    Event_data.append([Label[i] for i in range(0,len(Label))])
                    if(os.path.exists(save_dir + '/' + Name + str(count) + ".csv")):
                        os.remove(save_dir + '/' + Name + str(count) + ".csv")
                    writer = csv.writer(open(save_dir + '/' + Name + str(count) + ".csv", "a"))
                    writer.writerows(Event_data)


def MovieLabelDataSet(image_dir, seg_image_dir, csv_dir, save_dir, static_name, static_label, csv_name_diff, crop_size, gridx = 1, gridy = 1, offset = 0, yolo_v0 = False, 
yolo_v1 = True, yolo_v2 = False,  tshift  = 1, normalizeimage = True):
    
    
            raw_path = os.path.join(image_dir, '*tif')
            Seg_path = os.path.join(seg_image_dir, '*tif')
            Csv_path = os.path.join(csv_dir, '*csv')
            files_raw = glob.glob(raw_path)
            files_raw.sort
            filesSeg = glob.glob(Seg_path)
            filesSeg.sort
            filesCsv = glob.glob(Csv_path)
            filesCsv.sort
            Path(save_dir).mkdir(exist_ok=True)
            total_categories = len(static_name)
                
            
            
            for fname in files_raw:
                  
                 name = os.path.basename(os.path.splitext(fname)[0])   
                 for Segfname in filesSeg:
                      
                      Segname = os.path.basename(os.path.splitext(Segfname)[0])
                        
                      if name == Segname:
                          
                          
                         image = imread(fname)
                         if normalizeimage:
                            image = normalizeFloatZeroOne( image.astype('float32'),1,99.8)
                         segimage = imread(Segfname)
                        
                         for csvfname in filesCsv:
                                 count = 0  
                                 Csvname =  os.path.basename(os.path.splitext(csvfname)[0])
                                 for i in  range(0, len(static_name)):
                                     event_name = static_name[i]
                                     trainlabel = static_label[i]
                                     classfound = (Csvname == csv_name_diff +  event_name + name)   
                                     if classfound:
                                                    print(Csvname)
                                                    dataset = pd.read_csv(csvfname)
                                                    if len(dataset.keys()) >= 3:

                                                        time = dataset[dataset.keys()[0]][1:]
                                                        y = dataset[dataset.keys()[1]][1:]
                                                        x = dataset[dataset.keys()[2]][1:]
                                                        angle = np.full(time.shape, 2)                        
                                                    if len(dataset.keys()) > 3:

                                                        angle = dataset[dataset.keys()[3]][1:]                          
                                                    #Categories + XYHW + Confidence 
                                                    for (key, t) in time.items():
                                                       try: 
                                                          MovieMaker(t, y[key], x[key], angle[key], image, segimage, crop_size, gridx, gridy, offset, total_categories, trainlabel, name + event_name + str(count), save_dir,yolo_v0, yolo_v1, yolo_v2, tshift)
                                                          count = count + 1
                                                        
                                                       except:
                                                        
                                                           pass
                                                        

                                 
def MovieLabelDataSet4D(image_dir, seg_image_dir, csv_dir, save_dir, static_name, static_label, csv_name_diff, crop_size, gridx = 1, gridy = 1, offset = 0, yolo_v0 = False, 
yolo_v1 = True, yolo_v2 = False,  tshift  = 1, normalizeimage = True):
    
    
            raw_path = os.path.join(image_dir, '*tif')
            Seg_path = os.path.join(seg_image_dir, '*tif')
            Csv_path = os.path.join(csv_dir, '*csv')
            files_raw = glob.glob(raw_path)
            files_raw.sort
            filesSeg = glob.glob(Seg_path)
            filesSeg.sort
            filesCsv = glob.glob(Csv_path)
            filesCsv.sort
            Path(save_dir).mkdir(exist_ok=True)
            total_categories = len(static_name)
                
            
            
            for fname in files_raw:
                  
                 name = os.path.basename(os.path.splitext(fname)[0])   
                 for Segfname in filesSeg:
                      
                      Segname = os.path.basename(os.path.splitext(Segfname)[0])
                        
                      if name == Segname:
                         image = imread(fname)
                         
                         segimage = imread(Segfname)
                        
                         for csvfname in filesCsv:
                                 count = 0  
                                 Csvname =  os.path.basename(os.path.splitext(csvfname)[0])
                                 for i in  range(0, len(static_name)):
                                     event_name = static_name[i]
                                     trainlabel = static_label[i]
                                     classfound = (Csvname == csv_name_diff +  event_name + name)   
                                     if classfound:
                                                    print(Csvname)
                                                    dataset = pd.read_csv(csvfname)
                                                    if len(dataset.keys()) >= 3:

                                                        time = dataset[dataset.keys()[0]][1:]
                                                        z = dataset[dataset.keys()[1]][1:]
                                                        y = dataset[dataset.keys()[2]][1:]
                                                        x = dataset[dataset.keys()[3]][1:]
                                                        angle = np.full(time.shape, 2)                        
                                                    if len(dataset.keys()) > 4:

                                                        angle = dataset[dataset.keys()[4]][1:]                          
                                                    #Categories + XYHW + Confidence 
                                                    for (key, t) in time.items():
                                                       try: 
                                                          MovieMaker4D(normalizeimage, t, z[key], y[key], x[key], angle[key], image, segimage, crop_size, gridx, gridy, offset, total_categories, trainlabel, name + event_name + str(count), save_dir,yolo_v0, yolo_v1, yolo_v2, tshift)
                                                          count = count + 1
                                                        
                                                       except:
                                                        
                                                           pass                             

               

            
def MovieMaker(time, y, x, angle, image, segimage, crop_size, gridx, gridy, offset, total_categories, trainlabel, name, save_dir, yolo_v0, yolo_v1, yolo_v2, tshift):
    
       sizex, sizey, size_tminus, size_tplus = crop_size
       
       imagesizex = sizex * gridx
       imagesizey = sizey * gridy
       
       shiftNone = [0,0]
       if offset > 0 and trainlabel > 0:
                 shift_lx = [int(offset), 0] 
                 shift_rx = [-offset, 0]
                 shift_lxy = [int(offset), int(offset)]
                 shift_rxy = [-int(offset), int(offset)]
                 shift_dlxy = [int(offset), -int(offset)]
                 shift_drxy = [-int(offset), -int(offset)]
                 shift_uy = [0, int(offset)]
                 shift_dy = [0, -int(offset)]
                 AllShifts = [shiftNone, shift_lx, shift_rx,shift_lxy,shift_rxy,shift_dlxy,shift_drxy,shift_uy,shift_dy]

       else:
           
          AllShifts = [shiftNone]


       time = time - tshift
       if time > 0:
               currentsegimage = segimage[int(time),:].astype('uint16')
               height, width, center, seg_label = getHW(x, y, currentsegimage, imagesizex, imagesizey)
               for shift in AllShifts:

                        newname = name + 'shift' + str(shift)
                        Event_data = []
                        newcenter = (center[0] - shift[1],center[1] - shift[0] )
                        x = center[1]
                        y = center[0]
                        if yolo_v0:
                            Label = np.zeros([total_categories + 5])
                        if yolo_v1:    
                            Label = np.zeros([total_categories + 6])
                        if yolo_v2:
                            Label = np.zeros([total_categories + 7])
                        Label[trainlabel] = 1
                        #T co ordinate
                        Label[total_categories + 2] = (size_tminus) / (size_tminus + size_tplus)
                        if x + shift[0]> sizex/2 and y + shift[1] > sizey/2 and x + shift[0] + int(imagesizex/2) < image.shape[2] and y + shift[1]+ int(imagesizey/2) < image.shape[1] and time > size_tminus and time + size_tplus + 1 < image.shape[0]:
                                        crop_xminus = x  - int(imagesizex/2)
                                        crop_xplus = x  + int(imagesizex/2)
                                        crop_yminus = y  - int(imagesizey/2)
                                        crop_yplus = y   + int(imagesizey/2)
                                        region =(slice(int(time - size_tminus),int(time + size_tplus  + 1)),slice(int(crop_yminus)+ shift[1], int(crop_yplus)+ shift[1]),
                                              slice(int(crop_xminus) + shift[0], int(crop_xplus) + shift[0]))
                                        #Define the movie region volume that was cut
                                        crop_image = image[region]   

                                        seglocationx = (newcenter[1] - crop_xminus)
                                        seglocationy = (newcenter[0] - crop_yminus)

                                        Label[total_categories] =  seglocationx/sizex
                                        Label[total_categories + 1] = seglocationy/sizey
                                        if height >= imagesizey:
                                                        height = 0.5 * imagesizey
                                        if width >= imagesizex:
                                                        width = 0.5 * imagesizex
                                        #Height
                                        Label[total_categories + 3] = height/imagesizey
                                        #Width
                                        Label[total_categories + 4] = width/imagesizex

                                        if yolo_v1:
                                                  Label[total_categories + 5] = 1 
                                                 
                                        if yolo_v2:
                                             Label[total_categories + 5] = 1 
                                             Label[total_categories + 6] = angle        
                                        #Write the image as 32 bit tif file 
                                        if(crop_image.shape[0] == size_tplus + size_tminus + 1 and crop_image.shape[1]== imagesizey and crop_image.shape[2]== imagesizex):

                                                   imwrite((save_dir + '/' + newname + '.tif'  ) , crop_image.astype('float32'))    
                                                   Event_data.append([Label[i] for i in range(0,len(Label))])
                                                   if(os.path.exists(save_dir + '/' + (newname) + ".csv")):
                                                                os.remove(save_dir + '/' + (newname) + ".csv")
                                                   writer = csv.writer(open(save_dir + '/' + (newname) + ".csv", "a"))
                                                   writer.writerows(Event_data)
       
def MovieMaker4D(normalizeimage, time, z, y, x, angle, image, segimage, crop_size, gridx, gridy, offset, total_categories, trainlabel, name, save_dir, yolo_v0, yolo_v1, yolo_v2, tshift):
    
       sizex, sizey, size_tminus, size_tplus = crop_size
       
       imagesizex = sizex * gridx
       imagesizey = sizey * gridy
       
       shiftNone = [0,0]
       if offset > 0 and trainlabel > 0:
                 shift_lx = [int(offset), 0] 
                 shift_rx = [-offset, 0]
                 shift_lxy = [int(offset), int(offset)]
                 shift_rxy = [-int(offset), int(offset)]
                 shift_dlxy = [int(offset), -int(offset)]
                 shift_drxy = [-int(offset), -int(offset)]
                 shift_uy = [0, int(offset)]
                 shift_dy = [0, -int(offset)]
                 AllShifts = [shiftNone, shift_lx, shift_rx,shift_lxy,shift_rxy,shift_dlxy,shift_drxy,shift_uy,shift_dy]

       else:
           
          AllShifts = [shiftNone]


       time = time - tshift
       image = image[:,z,:,:]
       segimage = segimage[:,z,:,:]
       if normalizeimage:
                image = normalizeFloatZeroOne( image.astype('float32'),1,99.8)
       if time > 0:

               #slice the images
                

               currentsegimage = segimage[int(time),:,:].astype('uint16')
               

               height, width, center, seg_label = getHW(x, y, currentsegimage, imagesizex, imagesizey)
               for shift in AllShifts:

                        newname = name + 'shift' + str(shift)
                        Event_data = []
                        newcenter = (center[0] - shift[1],center[1] - shift[0] )
                        x = center[1]
                        y = center[0]
                        if yolo_v0:
                            Label = np.zeros([total_categories + 5])
                        if yolo_v1:    
                            Label = np.zeros([total_categories + 6])
                        if yolo_v2:
                            Label = np.zeros([total_categories + 7])
                        Label[trainlabel] = 1
                        #T co ordinate
                        Label[total_categories + 2] = (size_tminus) / (size_tminus + size_tplus)
                        if x + shift[0]> sizex/2 and y + shift[1] > sizey/2 and x + shift[0] + int(imagesizex/2) < image.shape[2] and y + shift[1]+ int(imagesizey/2) < image.shape[1] and time > size_tminus and time + size_tplus + 1 < image.shape[0]:
                                        crop_xminus = x  - int(imagesizex/2)
                                        crop_xplus = x  + int(imagesizex/2)
                                        crop_yminus = y  - int(imagesizey/2)
                                        crop_yplus = y   + int(imagesizey/2)
                                        region =(slice(int(time - size_tminus),int(time + size_tplus  + 1)),slice(int(crop_yminus)+ shift[1], int(crop_yplus)+ shift[1]),
                                              slice(int(crop_xminus) + shift[0], int(crop_xplus) + shift[0]))
                                        #Define the movie region volume that was cut
                                        crop_image = image[region]   

                                        seglocationx = (newcenter[1] - crop_xminus)
                                        seglocationy = (newcenter[0] - crop_yminus)

                                        Label[total_categories] =  seglocationx/sizex
                                        Label[total_categories + 1] = seglocationy/sizey
                                        if height >= imagesizey:
                                                        height = 0.5 * imagesizey
                                        if width >= imagesizex:
                                                        width = 0.5 * imagesizex
                                        #Height
                                        Label[total_categories + 3] = height/imagesizey
                                        #Width
                                        Label[total_categories + 4] = width/imagesizex



                                        if yolo_v1:
                                                  Label[total_categories + 5] = 1 
                                                 
                                        if yolo_v2:

                                             Label[total_categories + 5] = 1 
                                             Label[total_categories + 6] = angle        
                                        #Write the image as 32 bit tif file 
                                        if(crop_image.shape[0] == size_tplus + size_tminus + 1 and crop_image.shape[1]== imagesizey and crop_image.shape[2]== imagesizex):

                                                   imwrite((save_dir + '/' + newname + '.tif'  ) , crop_image.astype('float32'))    
                                                   Event_data.append([Label[i] for i in range(0,len(Label))])
                                                   if(os.path.exists(save_dir + '/' + (newname) + ".csv")):
                                                                os.remove(save_dir + '/' + (newname) + ".csv")
                                                   writer = csv.writer(open(save_dir + '/' + (newname) + ".csv", "a"))
                                                   writer.writerows(Event_data)


def Readname(fname):
    
    return os.path.basename(os.path.splitext(fname)[0])


def ImageLabelDataSet(image_dir, seg_image_dir, csv_dir,save_dir, static_name, static_label, csv_name_diff,crop_size, gridx = 1, gridy = 1, offset = 0, yolo_v0 = True, tshift  = 1):
    
    
            raw_path = os.path.join(image_dir, '*tif')
            Seg_path = os.path.join(seg_image_dir, '*tif')
            Csv_path = os.path.join(csv_dir, '*csv')
            files_raw = glob.glob(raw_path)
            files_raw.sort
            filesSeg = glob.glob(Seg_path)
            filesSeg.sort
            filesCsv = glob.glob(Csv_path)
            filesCsv.sort
            Path(save_dir).mkdir(exist_ok=True)
            total_categories = len(static_name)
            
            for csvfname in filesCsv:
              print(csvfname)
              count = 0
              Csvname =  os.path.basename(os.path.splitext(csvfname)[0])
            
              for fname in files_raw:
                  
                 name = os.path.basename(os.path.splitext(fname)[0])   
                 for Segfname in filesSeg:
                      
                      Segname = os.path.basename(os.path.splitext(Segfname)[0])
                        
                      if name == Segname:
                          
                          
                         image = imread(fname)
                         image = normalizeFloatZeroOne( image.astype('float32'),1,99.8)   
                         segimage = imread(Segfname)
                         for i in  range(0, len(static_name)):
                             event_name = static_name[i]
                             trainlabel = static_label[i]
                             if Csvname == csv_name_diff + name + event_name:
                                            dataset = pd.read_csv(csvfname)
                                            time = dataset[dataset.keys()[0]][1:]
                                            y = dataset[dataset.keys()[1]][1:]
                                            x = dataset[dataset.keys()[2]][1:]     
                                            
                                            #Categories + XYHW + Confidence 
                                            for (key, t) in time.items():
                                               ImageMaker(t, y[key], x[key], image, segimage, crop_size, gridx, gridy, offset, total_categories, trainlabel, name + event_name + str(count), save_dir,yolo_v0, tshift)    
                                               count = count + 1
    


    
def SegFreeImageLabelDataSet(image_dir, csv_dir,save_dir, static_name, static_label, csv_name_diff,crop_size, gridx = 1, gridy = 1, offset = 0):
    
    
            raw_path = os.path.join(image_dir, '*tif')
            Csv_path = os.path.join(csv_dir, '*csv')
            files_raw = glob.glob(raw_path)
            files_raw.sort
            filesCsv = glob.glob(Csv_path)
            filesCsv.sort
            Path(save_dir).mkdir(exist_ok=True)
            total_categories = len(static_name)
            
            for csvfname in filesCsv:
              print(csvfname)
              count = 0
              Csvname =  os.path.basename(os.path.splitext(csvfname)[0])
            
              for fname in files_raw:
                  
                         name = os.path.basename(os.path.splitext(fname)[0])   
                         image = imread(fname)
                       
                         image = normalizeFloatZeroOne( image.astype('float32'),1,99.8)   
                         for i in  range(0, len(static_name)):
                             event_name = static_name[i]
                             trainlabel = static_label[i]
                             if Csvname == csv_name_diff + name + event_name:
                                            dataset = pd.read_csv(csvfname)
                                            time = dataset[dataset.keys()[0]][1:]
                                            y = dataset[dataset.keys()[1]][1:]
                                            x = dataset[dataset.keys()[2]][1:]     
                                            
                                            #Categories + XYHW + Confidence 
                                            for (key, t) in time.items():
                                               SegFreeImageMaker(t, y[key], x[key], image, crop_size, gridx, gridy, offset, total_categories, trainlabel, name + event_name + str(count), save_dir)    
                                               count = count + 1                 
    
def createNPZ(save_dir, axes, save_name = 'Yolov0oneat', save_name_val = 'Yolov0oneatVal', expand = True, 
static = False, flip_channel_axis = False, train_size = 0.95):
            
            data = []
            label = []   
             
            raw_path = os.path.join(save_dir, '*tif')
            files_raw = glob.glob(raw_path)
            files_raw.sort
            NormalizeImages= [imread(fname) for fname in files_raw]
            
            names = [Readname(fname)  for fname in files_raw]
            #Normalize everything before it goes inside the training
            for i in range(0,len(NormalizeImages)):
                   
                       n = NormalizeImages[i]
                   
                       blankX = n
                       csvfname = save_dir + '/' + names[i] + '.csv'   
                       arr = [] 
                       with open(csvfname) as csvfile:
                             reader = csv.reader(csvfile, delimiter = ',')
                             arr =  list(reader)[0]
                             arr = np.array(arr)
                            
                       blankY = arr
                       blankY = np.expand_dims(blankY, -1)
                       if expand:
                         
                         blankX = np.expand_dims(blankX, -1)
                     
                       data.append(blankX)
                       label.append(blankY)



            dataarr = np.asarray(data)
            labelarr = np.asarray(label)
            if flip_channel_axis:
                       np.swapaxes(dataarr, 1,-1)
            if static:
                try:
                    
                   dataarr = dataarr[:,0,:,:,:]
                except:
                    
                    pass
            print(dataarr.shape, labelarr.shape)
            traindata, validdata, trainlabel, validlabel = train_test_split(dataarr, labelarr, train_size = train_size,
            test_size = 1 - train_size, shuffle = True)
            save_full_training_data(save_dir, save_name, traindata, trainlabel, axes)
            save_full_training_data(save_dir, save_name_val, validdata, validlabel, axes)


def _raise(e):
    raise e
def  ImageMaker(time, y, x, image, segimage, crop_size, gridX, gridY, offset, total_categories, trainlabel, name, save_dir, yolo_v0, tshift):

               sizeX, sizeY = crop_size

               ImagesizeX = sizeX * gridX
               ImagesizeY = sizeY * gridY

               shiftNone = [0,0]
               if offset > 0 and trainlabel > 0:
                         shiftLX = [int(offset), 0] 
                         shiftRX = [-offset, 0]
                         shiftLXY = [int(offset), int(offset)]
                         shiftRXY = [-int(offset), int(offset)]
                         shiftDLXY = [int(offset), -int(offset)]
                         shiftDRXY = [-int(offset), -int(offset)]
                         shiftUY = [0, int(offset)]
                         shiftDY = [0, -int(offset)]
                         AllShifts = [shiftNone, shiftLX, shiftRX,shiftLXY,shiftRXY,shiftDLXY,shiftDRXY,shiftUY,shiftDY]

               else:

                  AllShifts = [shiftNone]

               time = time - tshift
               if time < segimage.shape[0] - 1 and time > 0:
                 currentsegimage = segimage[int(time),:].astype('uint16')
                
                 height, width, center, seg_label  = getHW(x, y, currentsegimage, ImagesizeX, ImagesizeY)
                 for shift in AllShifts:
                   
                        newname = name + 'shift' + str(shift)
                        newcenter = (center[0] - shift[1],center[1] - shift[0] )
                        Event_data = []
                        
                        x = center[1]
                        y = center[0]
                        if yolo_v0:
                          Label = np.zeros([total_categories + 4])
                        else:
                          Label = np.zeros([total_categories + 5])  
                        Label[trainlabel] = 1
                        if x + shift[0]> sizeX/2 and y + shift[1] > sizeY/2 and x + shift[0] + int(ImagesizeX/2) < image.shape[2] and y + shift[1]+ int(ImagesizeY/2) < image.shape[1]:
                                    crop_Xminus = x  - int(ImagesizeX/2)
                                    crop_Xplus = x   + int(ImagesizeX/2)
                                    crop_Yminus = y  - int(ImagesizeY/2)
                                    crop_Yplus = y   + int(ImagesizeY/2)
                                    
                                    for tex in range(int(time) -2, int(time) + 2):
                                                    newname = newname + str(tex)
                                                    region =(slice(int(tex - 1),int(tex)),slice(int(crop_Yminus)+ shift[1], int(crop_Yplus)+ shift[1]),
                                                           slice(int(crop_Xminus) + shift[0], int(crop_Xplus) + shift[0]))

                                                    crop_image = image[region]      


                                                    seglocationx = (newcenter[1] - crop_Xminus)
                                                    seglocationy = (newcenter[0] - crop_Yminus)

                                                    Label[total_categories] =  seglocationx/sizeX
                                                    Label[total_categories + 1] = seglocationy/sizeY

                                                    if height >= ImagesizeY:
                                                        height = 0.5 * ImagesizeY
                                                    if width >= ImagesizeX:
                                                        width = 0.5 * ImagesizeX

                                                    Label[total_categories + 2] = height/ImagesizeY
                                                    Label[total_categories + 3] = width/ImagesizeX
                                                    if yolo_v0==False:
                                                            Label[total_categories + 4] = 1 

                                                    if(crop_image.shape[1]== ImagesizeY and crop_image.shape[2]== ImagesizeX):
                                                             imwrite((save_dir + '/' + newname + '.tif'  ) , crop_image.astype('float32'))  
                                                             Event_data.append([Label[i] for i in range(0,len(Label))])
                                                             if(os.path.exists(save_dir + '/' + (newname) + ".csv")):
                                                                os.remove(save_dir + '/' + (newname) + ".csv")
                                                             writer = csv.writer(open(save_dir + '/' + (newname) + ".csv", "a"))
                                                             writer.writerows(Event_data)

def  SegFreeImageMaker(time, y, x, image, crop_size, gridX, gridY, offset, total_categories, trainlabel, name, save_dir, yolo_v0, tshift):

               sizex, sizey = crop_size

               ImagesizeX = sizex * gridX
               ImagesizeY = sizey * gridY

               shiftNone = [0,0]
               if offset > 0 and trainlabel > 0:
                         shiftLX = [int(offset), 0] 
                         shiftRX = [-offset, 0]
                         shiftLXY = [int(offset), int(offset)]
                         shiftRXY = [-int(offset), int(offset)]
                         shiftDLXY = [int(offset), -int(offset)]
                         shiftDRXY = [-int(offset), -int(offset)]
                         shiftUY = [0, int(offset)]
                         shiftDY = [0, -int(offset)]
                         AllShifts = [shiftNone, shiftLX, shiftRX,shiftLXY,shiftRXY,shiftDLXY,shiftDRXY,shiftUY,shiftDY]

               else:

                  AllShifts = [shiftNone]

               time = time - tshift
               if time < image.shape[0] - 1 and time > 0:
                
                 for shift in AllShifts:
                   
                        newname = name + 'shift' + str(shift)
                        newcenter = (y - shift[1],x - shift[0] )
                        Event_data = []
                        
                        if yolo_v0:
                          Label = np.zeros([total_categories + 4])
                        else:
                          Label = np.zeros([total_categories + 5])  
                        Label[trainlabel] = 1
                        if x + shift[0]> sizex/2 and y + shift[1] > sizey/2 and x + shift[0] + int(ImagesizeX/2) < image.shape[2] and y + shift[1]+ int(ImagesizeY/2) < image.shape[1]:
                                    crop_Xminus = x  - int(ImagesizeX/2)
                                    crop_Xplus = x   + int(ImagesizeX/2)
                                    crop_Yminus = y  - int(ImagesizeY/2)
                                    crop_Yplus = y   + int(ImagesizeY/2)
                                    
                                    for tex in range(int(time) -2, int(time) + 2):
                                                    newname = newname + str(tex)
                                                    region =(slice(int(tex - 1),int(tex)),slice(int(crop_Yminus)+ shift[1], int(crop_Yplus)+ shift[1]),
                                                           slice(int(crop_Xminus) + shift[0], int(crop_Xplus) + shift[0]))

                                                    crop_image = image[region]      


                                                    seglocationx = (newcenter[1] - crop_Xminus)
                                                    seglocationy = (newcenter[0] - crop_Yminus)

                                                    Label[total_categories] =  seglocationx/sizex
                                                    Label[total_categories + 1] = seglocationy/sizey
                                                    Label[total_categories + 2] = 1
                                                    Label[total_categories + 3] = 1
                                                    if yolo_v0==False:
                                                            Label[total_categories + 4] = 1 

                                                    if(crop_image.shape[1]== ImagesizeY and crop_image.shape[2]== ImagesizeX):
                                                             imwrite((save_dir + '/' + newname + '.tif'  ) , crop_image.astype('float32'))  
                                                             Event_data.append([Label[i] for i in range(0,len(Label))])
                                                             if(os.path.exists(save_dir + '/' + (newname) + ".csv")):
                                                                os.remove(save_dir + '/' + (newname) + ".csv")
                                                             writer = csv.writer(open(save_dir + '/' + (newname) + ".csv", "a"))
                                                             writer.writerows(Event_data)



       
def getHW(defaultX, defaultY, currentsegimage, imagesizex, imagesizey):
    
    properties = measure.regionprops(currentsegimage, currentsegimage)
    TwoDLocation = (defaultY,defaultX)
    SegLabel = currentsegimage[int(TwoDLocation[0]), int(TwoDLocation[1])]
    for prop in properties:
                                               
                  if SegLabel > 0 and prop.label == SegLabel:
                                    minr, minc, maxr, maxc = prop.bbox
                                    center = (defaultY, defaultX)
                                    height =  abs(maxc - minc)
                                    width =  abs(maxr - minr)
                                
                  if SegLabel == 0 :
                    
                             center = (defaultY, defaultX)
                             height = 0.5 * imagesizex
                             width = 0.5 * imagesizey
                               
                    
                                
    return height, width, center, SegLabel     

def save_full_training_data(directory, filename, data, label, axes):
    """Save training data in ``.npz`` format."""
  

    len(axes) == data.ndim or _raise(ValueError())
    np.savez(directory + filename, data = data, label = label, axes = axes)
    
def InterchangeTXY(TXYCSV, save_dir):

    
     dataset = pd.read_csv(TXYCSV)
     time = dataset[dataset.keys()[0]][1:]
     x = dataset[dataset.keys()[1]][1:]
     y = dataset[dataset.keys()[2]][1:]
 
     Event_data = []
     
     Name = os.path.basename(os.path.splitext(TXYCSV)[0])

     for (key, t) in time.items():
         
         Event_data.append([t, y[key], x[key]])
         
     writer = csv.writer(open(save_dir + '/' + (Name) + ".csv", "a"))
     writer.writerows(Event_data)    
         
    
def  AngleAppender(AngleCSV, ONTCSV, save_dir, ColumnA = 'Y'):

     dataset = pd.read_csv(AngleCSV)
     time = dataset[dataset.keys()[0]][1:]
     if ColumnA == 'Y':
       y = dataset[dataset.keys()[1]][1:]
       x = dataset[dataset.keys()[2]][1:]
       angle = dataset[dataset.keys()[3]][1:]
     else:
       x = dataset[dataset.keys()[1]][1:]
       y = dataset[dataset.keys()[2]][1:]
       angle = dataset[dataset.keys()[3]][1:]  
       
     clickeddataset = pd.read_csv(ONTCSV)  
     
     clickedtime = clickeddataset[clickeddataset.keys()[0]][1:]
     clickedy = clickeddataset[clickeddataset.keys()[1]][1:]
     clickedx = clickeddataset[clickeddataset.keys()[2]][1:]
                              
     Event_data = []
     
     Name = os.path.basename(os.path.splitext(ONTCSV)[0])
     

     
     for (clickedkey, clickedt) in clickedtime.items():
                
                       
                for (key, t) in time.items(): 
                         
                          if t == clickedt and y[key] == clickedy[clickedkey] and x[key] == clickedx[clickedkey]:
                              
                                Event_data.append([t,y[key],x[key],angle[key]])
                              
                                     
                              
                               
     writer = csv.writer(open(save_dir + '/' + (Name) + ".csv", "a"))
     writer.writerows(Event_data)
         
