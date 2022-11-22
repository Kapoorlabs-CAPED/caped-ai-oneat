from napari.viewer import Viewer
from tifffile import imread
import numpy as np
import pandas as pd
class OneatSimpleVisualization:
    
    def __init__(self, viewer: Viewer,
                 imagename: str, 
                 csvname: str):
        
        self.viewer = viewer 
        self.imagename = imagename 
        self.csvname = csvname
        
        
    def show_image(self):
        
        name_remove = ("Image")
        for layer in list(self.viewer.layers):
            if any(name in layer.name for name in name_remove ):
                self.viewer.layers.remove(layer)
        
        
        self.image = imread(self.imagename)   
        self.viewer.add_image(self.image, name ="Image" + self.imagename )
        
  
                
    def show_csv(self, event_threshold):
                   
            event_locations_dict = {}
            event_locations_size_dict = {}
            event_locations = []     
            size_locations = [] 
            score_locations = []
            confidence_locations = [] 
            detections = pd.read_csv(self.csvname, delimiter=',')
            nrows = len(detections.columns)
            for index, row in detections.iterrows():
                tcenter = int(row[0])
                zcenter = row[1]
                ycenter = row[2]
                xcenter = row[3]
                if nrows > 4:
                    score = row[4]
                    size = row[5]
                    confidence = row[6]
                else:
                    score = 1.0
                    size = 10
                    confidence = 1.0
                if score > event_threshold:
                    event_locations.append(
                        [
                            int(tcenter),
                            int(zcenter),
                            int(ycenter),
                            int(xcenter),
                        ]
                    )

                    if int(tcenter) in event_locations_dict.keys():
                        current_list = event_locations_dict[int(tcenter)]
                        current_list.append(
                            [int(zcenter), int(ycenter), int(xcenter)]
                        )
                        event_locations_dict[int(tcenter)] = current_list
                        event_locations_size_dict[
                            (
                                int(tcenter),
                                int(zcenter),
                                int(ycenter),
                                int(xcenter),
                            )
                        ] = [size, score, confidence]
                    else:
                        current_list = []
                        current_list.append(
                            [int(zcenter), int(ycenter), int(xcenter)]
                        )
                        event_locations_dict[int(tcenter)] = current_list
                        event_locations_size_dict[
                            int(tcenter),
                            int(zcenter),
                            int(ycenter),
                            int(xcenter),
                        ] = [size, score, confidence]

                    size_locations.append(size)
                    score_locations.append(score)
                    confidence_locations.append(confidence)
            point_properties = {
                "score": np.array(self.score_locations),
                "confidence": np.array(self.confidence_locations),
                "size": np.array(self.size_locations),
            }

            name_remove = ("Detections", "Location Map")
            for layer in list(self.viewer.layers):

                if any(name in layer.name for name in name_remove):
                    self.viewer.layers.remove(layer)
            if len(self.score_locations) > 0:
                self.viewer.add_points(
                    self.event_locations,
                    properties=point_properties,
                    symbol="square",
                    blending="translucent_no_depth",
                    name="Detections",
                    face_color=[0] * 4,
                    edge_color="red",
                )
           
           
                     
                        