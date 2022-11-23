from napari.viewer import Viewer
from tifffile import imread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from  napari import layers
class OneatSimpleVisualization:
    
    def __init__(self, viewer: Viewer,
                 ax,
                 figure: plt):
        
        self.viewer = viewer 
        self.ax = ax
        self.figure = figure
        
        
    def show_csv(self, csvname, event_threshold):
                   
            self.ax.cla()       
            for layer in list(self.viewer.layers):
               if isinstance(layer, layers.Image):
                  self.image = layer.data
        
            self.totaltime = self.image.shape[0]
                
            event_locations_dict = {}
            event_locations_size_dict = {}
            event_locations = []     
            size_locations = [] 
            score_locations = []
            confidence_locations = [] 
            current_list = []
            detections = pd.read_csv(csvname, delimiter=',')
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
                "score": np.array(score_locations),
                "confidence": np.array(confidence_locations),
                "size": np.array(size_locations),
            }

            name_remove = ("Detections", "Location Map")
            for layer in list(self.viewer.layers):

                if any(name in layer.name for name in name_remove):
                    self.viewer.layers.remove(layer)
            if len(score_locations) > 0:
                self.viewer.add_points(
                    event_locations,
                    properties=point_properties,
                    symbol="square",
                    blending="translucent_no_depth",
                    name="Detections",
                    face_color=[0] * 4,
                    edge_color="red",
                )
                
            timelist = []
            countlist = []    
            for tcenter in range(self.totaltime):
                
                timelist.append(tcenter)
                if int(tcenter) in event_locations_dict:
                  countlist.append(len(event_locations_dict[int(tcenter)]))
                else:
                    countlist.append(0)  
                
             
            self.ax.plot(timelist, countlist, "-g")
            self.ax.set_title("Events")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Counts")
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

            self.figure.savefig(
                 str(Path(csvname).parent.as_posix())
                + str('visualplot')
                + str('.png'),
                  dpi=300
            )         
                        