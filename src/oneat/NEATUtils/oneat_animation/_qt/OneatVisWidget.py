from qtpy.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QDoubleSpinBox, QSlider, QLabel, QPushButton
)
from qtpy.QtCore import Qt

import matplotlib.pyplot as plt 
from matplotlib.backends.backend_qt5agg import ( FigureCanvasQTAgg as FigureCanvas )

class OneatVisWidget(QWidget):
    
    def __init__(self, parent = None):
        
        super.__init__(parent = parent)
        
        self._layout = QFormLayout(parent= self)
        
        
        
        self.imageidbox = QComboBox()
        index = self.imageidbox.findText('linear', Qt.MatchFixedString)
        self.imageidbox.setCurrentIndex(index)
        
        self.detectionidbox = QComboBox()
        index = self.detectionidbox.findText('linear', Qt.MatchFixedString)
        self.detectionidbox.setCurrentIndex(index)
        
        
        
        self.startprobspinbox = QDoubleSpinBox()
        self.startprobspinbox.setValue(0.9)
        self.startprobspinbox.setDecimals(10)
        
        self.scoreslider = QSlider(Qt.Horizontal, parent = self)
        self.scoreslider.setToolTip('Scroll through probability score')
        self.scoreslider.setRange(5000)
        self.scoreslider.singleStep(1)
        self.scoreslider.tickInterval(1)
        self.scoreslider.setValue(0)
        
        self.label = QLabel(parent= self)
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.label.setMinimumWidth(80)
        self.label.setText(f"{0.9:.5f}")
        
        self.recomputebutton = QPushButton("Recompute with changed parameters",parent= self)
        
        
        self.figure = plt.figure(figsize=(4, 4))
        self.multiplot_widget = FigureCanvas(self.figure)
        self.multiplot_widget.setMinimumSize(200, 200)
        self.ax = self.multiplot_widget.figure.subplots(1, 1)

        self._layout.addWidget(self.multiplot_widget)
        self._layout.addRow("Image",self.imageidbox)
        self._layout.addRow("Detections", self.detectionidbox)
        self._layout.addRow("Lowest probability event", self.startprobspinbox)
        self._layout.addRow("Score slider", self.scoreslider)
        self._layout.addRow(self.recomputebutton)
        
        
        
    
    