from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QWidget, QFormLayout



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

        self._layout.addRow('Event', self.eventidbox)
        self._layout.addRow('Plot', self.plotidbox)
        self._layout.addRow('Image/Movie', self.imageidbox)
