from pathlib import Path
import json
from oneat.NEATUtils.oneat_animation._qt import OneatVisualizeWidget
import napari


class VizDet:
    
    def __init__(self,  csvfile: str):
        
        self.csvfile = csvfile
        
  
        
    def show_detections(self):
        
        self.viz_widget = OneatVisualizeWidget(napari.Viewer(), self.csvfile)
        
        dock_widget = self.viewer.window.add_dock_widget(
            self.viz_widget, area="right"
        )
        self.viewer.window._qt_window.resizeDocks(
            [dock_widget], [200], Qt.Horizontal
        )

        napari.run()      
        