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
        
        
        self.viswidget.startprobspinbox.connect(self._update_startprob_callback())
        self.viswidget.scoreslider.SliderValueChange.connect(self._update_slider_callback())
        
        self.simplevisualization = OneatSimpleVisualization(viewer, imagename, csvname, self.viswidget.ax, self.viswidget.figure )
        
        self.viswidget.imageidbox.currentIndexChanged.connect(
            lambda imageid = self.viswidget.imageidbox : self._capture_image_callback()
        )
        
        self.viswidget.detectionidbox.currentIndexChanged.connect(
            lambda detectioid = self.viswidget.detectionidbox: self._capture_detections_callback()
        )
    
    def _update_startprob_callback(self, event):
        self.start_prob = self.viswidget.startprobspinbox.value()    
        self.event_threshold = float(self.viswidget.label.text())

    def _update_slider_callback(self, value):

        real_value = float(
            self.start_prob + (1.0 - self.start_prob) / 5000 * float(value)
        )
        self.viswidget.label.setText(str(real_value))
        self.event_threshold = float(real_value)    
        
    def _capture_image_callback(self):
            
           
           self.simplevisualization.show_image()
           
           
    def _capture_detections_callback(self):  
        
          self.simplevisualization.show_csv(self.event_threshold)
                