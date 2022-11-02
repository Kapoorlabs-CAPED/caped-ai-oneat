import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QWidget,
)


class OneatFrameWidget(QWidget):
    """Widget for interatviely making animations using the napari viewer."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self._layout = QFormLayout(parent=self)
        self.eventidbox = QComboBox()
        index = self.eventidbox.findText("linear", Qt.MatchFixedString)
        self.eventidbox.setCurrentIndex(index)

        self.plotidbox = QComboBox()
        index = self.plotidbox.findText("linear", Qt.MatchFixedString)
        self.plotidbox.setCurrentIndex(index)

        self.imageidbox = QComboBox()
        index = self.imageidbox.findText("linear", Qt.MatchFixedString)
        self.imageidbox.setCurrentIndex(index)

        self.heatstepsSpinBox = QSpinBox()
        self.heatstepsSpinBox.setValue(1)
        self.heatstepsSpinBox.setMaximum(100000)
        self.startprobSpinBox = QDoubleSpinBox()

        self.nmsspaceSpinBox = QDoubleSpinBox()
        self.nmsspaceSpinBox.setValue(10)
        self.nmsspaceSpinBox.setMaximum(100000)

        self.startprobSpinBox.setValue(0.9)
        self.startprobSpinBox.setDecimals(10)

        self.scoreSlider = QSlider(Qt.Horizontal, parent=self)
        self.scoreSlider.setToolTip("Scroll through probability score")
        self.scoreSlider.setRange(0, 5000)
        self.scoreSlider.setSingleStep(1)
        self.scoreSlider.setTickInterval(1)
        self.scoreSlider.setValue(0)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.label.setMinimumWidth(80)
        self.label.setText(f"{0.9:.5f}")

        self.recomputeButton = QPushButton(
            "Recompute with changed parameters", parent=self
        )

        self.figure = plt.figure(figsize=(4, 4))
        self.multiplot_widget = FigureCanvas(self.figure)
        self.multiplot_widget.setMinimumSize(200, 200)
        self.ax = self.multiplot_widget.figure.subplots(1, 1)

        self._layout.addWidget(self.multiplot_widget)
        self._layout.addRow("Image/Movie", self.imageidbox)
        self._layout.addRow("Event", self.eventidbox)
        self._layout.addRow("Heat Map Steps", self.heatstepsSpinBox)
        self._layout.addRow("NMS space (px)", self.nmsspaceSpinBox)
        self._layout.addRow("Lowest probability event", self.startprobSpinBox)
        self._layout.addRow("Score slider", self.scoreSlider)
        self._layout.addRow("Score threshold", self.label)
        self._layout.addRow("Plot", self.plotidbox)
        self._layout.addRow(self.recomputeButton)
