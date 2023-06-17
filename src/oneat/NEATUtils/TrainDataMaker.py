import napari
import os

from tifffile import imread

import pandas as pd



class TrainDataMaker(object):
       def __init__(self, source_dir):
              
              self.source_dir = source_dir
              self.acceptable_formats = [".tif", ".TIFF", ".TIF", ".png"]
              self._click_maker()

       def _click_maker(self):
           from qtpy.QtWidgets import QComboBox, QPushButton

           self.viewer = napari.Viewer()
           Imageids = []
           Boxname = 'ImageIDBox'
           X = os.listdir(self.source_dir)
           imageidbox = QComboBox()   
           imageidbox.addItem(Boxname)   
           tracksavebutton = QPushButton('Save Clicks')
           for imagename in X:
                  if any(imagename.endswith(f) for f in self.acceptable_formats):
                      Imageids.append(os.path.join(self.source_dir,imagename))
           for i in range(0, len(Imageids)):
                 imageidbox.addItem(str(Imageids[i])) 

           self.viewer.window.add_dock_widget(imageidbox, name="Image", area='bottom')    
           self.viewer.window.add_dock_widget(tracksavebutton, name="Save Clicks", area='bottom')
           imageidbox.currentIndexChanged.connect(
                        lambda trackid = imageidbox: EventViewer(
                                self.viewer,
                                imageidbox.currentText(),
                                os.path.basename(os.path.splitext(imageidbox.currentText())[0]), self.source_dir, False, True ))     

           tracksavebutton.clicked.connect(
                        lambda trackid= tracksavebutton:EventViewer(
                                self.viewer,
                                imageidbox.currentText(),
                                os.path.basename(os.path.splitext(imageidbox.currentText())[0]), self.source_dir, True, False ))                
                     
           napari.run()   


class EventViewer(object):

    def __init__(self, viewer, imagename, Name, csv_dir, save = False, newimage = True):
     
          self.save = save
          self.newimage = newimage
          self.viewer = viewer
          print('reading image')      
          self.imagename = imagename  
          self.image = imread(imagename)
          print('image read')
          self.Name = Name
          self.ndim = len(self.image.shape)  
          self.csv_dir = csv_dir
          assert self.ndim == 4, f'Input image should be 4 dimensional, try Training_data_maker for 2D + time images'
          
          self._click()

    def _click(self):

        
                ClassBackground = 'Normal'
                ClassMitosis = 'Mitosis'
                ClassApoptosis = 'Apoptosis'
                
                if self.save == True:
 
                        ClassBackgrounddata = self.viewer.layers[ClassBackground].data
                        bgdf = pd.DataFrame(ClassBackgrounddata, index = None, columns = ['T', 'Z', 'Y', 'X'])
                        bgdf.to_csv(self.csv_dir + '/' + 'ONEAT' + ClassBackground + self.Name +  '.csv', index = False, mode = 'w')

                        ClassMitosisdata = self.viewer.layers[ClassMitosis].data
                        divdf = pd.DataFrame(ClassMitosisdata, index = None, columns = ['T', 'Z', 'Y', 'X'])
                        divdf.to_csv(self.csv_dir + '/' +'ONEAT' + ClassMitosis + self.Name +  '.csv', index = False, mode = 'w')

                        ClassApoptosisdata = self.viewer.layers[ClassApoptosis].data
                        apopdf = pd.DataFrame(ClassApoptosisdata, index = None, columns = ['T', 'Z', 'Y', 'X'])
                        apopdf.to_csv(self.csv_dir + '/' +'ONEAT' + ClassApoptosis + self.Name +  '.csv', index = False, mode = 'w')

                       
        
                if self.newimage == True:
                     for layer in list(self.viewer.layers):

                            self.viewer.layers.remove(layer) 
                    
                if self.save == False:
                        self.viewer.add_image(self.image, name = self.Name)


                        # add the first points layer for ClassBackground Z and T point
                        self.viewer.add_points(name= ClassBackground, face_color='red', ndim = 4)

                        # add the second points layer for ClassMitosis Z and T point
                        self.viewer.add_points(name= ClassMitosis, face_color='blue', ndim = 4)

                        # add the second points layer for ClassApoptosis Z and T point
                        self.viewer.add_points(name= ClassApoptosis, face_color='green', ndim = 4)
                      
                        # programatically enter add mode for both Points layers to enable editing

                        self.viewer.layers[ClassBackground].mode = 'add'
                        self.viewer.layers[ClassMitosis].mode = 'add'
                        self.viewer.layers[ClassApoptosis].mode = 'add'