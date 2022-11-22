from pathlib import Path
import json
from oneat.NEATUtils.oneat_animation._qt import OneatVisualizeWidget
import napari
from qtpy.QtCore import Qt

class VizDet:
    
        
        
    def __init__(self):
        
        self.viewer = napari.Viewer()
        self.viz_widget = OneatVisualizeWidget(self.viewer)
        
        dock_widget = self.viewer.window.add_dock_widget(
            self.viz_widget, area="right"
        )
        self.viewer.window._qt_window.resizeDocks(
            [dock_widget], [200], Qt.Horizontal
        )

        napari.run()      
        