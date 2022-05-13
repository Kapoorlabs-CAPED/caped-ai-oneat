from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QWidget, QFormLayout, QSpinBox, QSlider
from oneat.NEATUtils.napari_animation.easing import Easing
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class OneatFrameWidget(QWidget):
    """Widget for interatviely making animations using the napari viewer."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self._layout = QFormLayout(parent=self)
        self.eventidbox = QComboBox()
        index = self.eventidbox.findText('linear', Qt.MatchFixedString)
        self.eventidbox.setCurrentIndex(index)
        
        self.plotidbox = QComboBox()
        index = self.plotidbox.findText('linear', Qt.MatchFixedString)
        self.plotidbox.setCurrentIndex(index)

        self.imageidbox = QComboBox()
        index = self.imageidbox.findText('linear', Qt.MatchFixedString)
        self.imageidbox.setCurrentIndex(index)

        self.heatstepsSpinBox = QSpinBox()
        self.heatstepsSpinBox.setValue(1)

        self.scoreSlider = QSlider(Qt.Horizontal, parent=self)
        self.scoreSlider.setToolTip("Scroll through probability score")
        self.scoreSlider.setRange(0, 1)
        self.scoreSlider.setSingleStep(0.01)


        self.figure = plt.figure(figsize=(4, 4))
        self.multiplot_widget = FigureCanvas(self.figure)
        self.ax = self.multiplot_widget.self.figure.subplots(1, 1)
        

        self._layout.addWidget(self.multiplot_widget)
        self._layout.addRow('Event', self.eventidbox)
        self._layout.addRow('Image/Movie', self.imageidbox)
        self._layout.addRow('Plot', self.plotidbox)
        self._layout.addRow('Heat Map Steps', self.heatstepsSpinBox)
        self._layout.addRow('Score slider', self.scoreSlider)
        