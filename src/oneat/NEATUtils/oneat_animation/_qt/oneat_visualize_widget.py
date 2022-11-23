from napari.viewer import Viewer
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QLabel, QFileDialog)
from .OneatVisWidget import OneatVisWidget
from ..OneatSimpleVisualization import OneatSimpleVisualization

class OneatVisualizeWidget(QWidget):
    
    def __init__(self,
                 viewer : Viewer,
                 parent = None):
        
        super().__init__(parent = parent)
    
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        
        self._layout.addWidget(QLabel("Visualize Detections", parent = self))
        self.viswidget = OneatVisWidget(parent = self)
        self._layout.addWidget(self.viswidget)
        self.event_threshold = float(self.viswidget.label.text())
        self.start_prob = self.viswidget.startprobspinbox.value()
        self.viswidget.startprobspinbox.valueChanged.connect(self._update_startprob_callback)
        self.viswidget.scoreslider.valueChanged.connect(self._update_slider_callback)
        
        self.simplevisualization = OneatSimpleVisualization(viewer,  self.viswidget.ax, self.viswidget.figure )
        
       
        
        self.viswidget.detectionidbox.clicked.connect(
            lambda detectioid = self.viswidget.detectionidbox: self._capture_detections_callback()
        )
        self.viswidget.recomputebutton.clicked.connect(
            lambda eventid=self.viswidget.recomputebutton: self._recompute_callback()
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
        
   
    def _recompute_callback(self):  
        
          if self.csvname is not None:
                self.event_threshold = float(self.viswidget.label.text())
                self.simplevisualization.show_csv(self.csvname,self.event_threshold)       
           
    def _capture_detections_callback(self):  
        
          self.csvname = QFileDialog.getOpenFileName(self, "Open oneat detections file")
          if self.csvname is not None:
                self.csvname = self.csvname[0]
                self.event_threshold = float(self.viswidget.label.text())
                self.simplevisualization.show_csv(self.csvname,self.event_threshold)
                        