from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QWidget, QFormLayout, QSpinBox
from oneat.NEATUtils.napari_animation.easing import Easing


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

        self.stepsSpinBox = QSpinBox()
        self.stepsSpinBox.setValue(1)

        self.stepsSpinBox = QSpinBox()
        self.stepsSpinBox.setValue(1)

        self.startframeSpinBox = QSpinBox()
        self.startframeSpinBox.setValue(0)

        self.endframeSpinBox = QSpinBox()
        self.endframeSpinBox.setValue(10)

        self.easeComboBox = QComboBox()
        self.easeComboBox.addItems([e.name.lower() for e in Easing])
        index = self.easeComboBox.findText('linear', Qt.MatchFixedString)
        self.easeComboBox.setCurrentIndex(index)

        self._layout.addRow('Event', self.eventidbox)
        self._layout.addRow('Plot', self.plotidbox)
        self._layout.addRow('Image/Movie', self.imageidbox)
        self._layout.addRow('Heat Map Steps', self.stepsSpinBox)
        self._layout.addRow('Steps', self.stepsSpinBox)
        self._layout.addRow('Ease', self.easeComboBox)
        self._layout.addRow('StartFrame', self.startframeSpinBox)
        self._layout.addRow('EndFrame', self.endframeSpinBox)