
from napari.viewer import Viewer
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QLabel)
from .OneatVisWidget import OneatVisWidget
from ..OneatSimpleVisualization import OneatSimpleVisualization

class oneat_visualize_widget(QWidget):
    
    def __init__(self,
                 viewer : Viewer,
                 imagename : str, 
                 csvname: str, 
                 parent = None):
        
        super().__init__(parent = parent)
    
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        
        self._layout.addWidget(QLabel("Visualize Detections", parent = self))
        self.viswidget = OneatVisWidget(parent= self)
        self._layout.addWidget(self.viswidget)
        
        self.simplevisualization = OneatSimpleVisualization(viewer, imagename, csvname )
        
        self.viswidget.imageidbox.currentIndexChanged.connect(
            lambda imageid = self.viswidget.imageidbox : self._capture_image_callback(imagename)
        )
        
    def _capture_image_callback(self, imagename):
            
           get_image_text = self.viswidget.imageidbox.currentText() 
           
           self.simplevisualization.show_image()