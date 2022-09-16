from qtpy.QtWidgets import QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton
from .OneatFrameWidget import OneatFrameWidget
import os
import napari
from dask.array.image import imread as daskread
from tifffile import imread,  imwrite
from ..OneatVolumeVisualization import OneatVolumeVisualization
from napari_animation._qt import AnimationWidget

class OneatVolumeWidget(QWidget):
    """Widget for interatviely making oneat visualizations using the napari viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        napari viewer.

    Attributes
    ----------
    key_frames : list of dict
        List of viewer state dictionaries.
    frame : int
        Currently shown key frame.
    """

    def __init__(
        self,
        viewer: 'napari.viewer.Viewer',
        csvdir: None,
        savedir: None,
        savename: None,
        key_categories : None,
        use_dask = False,
        segimagedir = None,
        heatimagedir = None,
        heatname = '_Heat',
        event_count_plot = 'Plot selected event count',
        parent=None,
    ):
        super().__init__(parent=parent)

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self._layout.addWidget(QLabel('Oneat Visualization Wizard', parent=self))
        
        self.frameWidget = OneatFrameWidget(parent=self)
        self._layout.addWidget(self.frameWidget)
        self._layout.addStretch(1)

        animation = AnimationWidget(viewer)
        self.start_prob = self.frameWidget.startprobSpinBox.value()
        self.nms_space = self.frameWidget.nmsspaceSpinBox.value()

        self._layout.addWidget(animation)
        self.oneatvisualization = OneatVolumeVisualization(viewer ,key_categories, csvdir, savedir, savename, self.frameWidget.ax, self.frameWidget.figure)
       
        self.heatmapsteps = self.frameWidget.heatstepsSpinBox.value()
        self.event_threshold = float(self.frameWidget.label.text())


        self.frameWidget.plotidbox.addItem('Select a type of plot')
        self.frameWidget.plotidbox.addItem(event_count_plot)
        self.frameWidget.heatstepsSpinBox.valueChanged.connect(self.update_heat_steps)
        self.frameWidget.startprobSpinBox.valueChanged.connect(self.update_start_prob)
        self.frameWidget.nmsspaceSpinBox.valueChanged.connect(self.update_nms_space)

        self.frameWidget.scoreSlider.valueChanged.connect(self.updateLabel)
    

        
        self.frameWidget.imageidbox.currentIndexChanged.connect(lambda eventid = self.frameWidget.imageidbox :
        self._capture_image_callback(segimagedir, heatimagedir, heatname,  use_dask ))

        self.frameWidget.eventidbox.currentIndexChanged.connect(lambda eventid = self.frameWidget.eventidbox :
        self._capture_csv_callback(segimagedir,  use_dask ))
 
        self.frameWidget.plotidbox.currentIndexChanged.connect(lambda eventid = self.frameWidget.imageidbox :
        self._capture_plot_callback(segimagedir,event_count_plot))

        self.frameWidget.recomputeButton.clicked.connect(lambda eventid = self.frameWidget.recomputeButton :
        self._start_callbacks(segimagedir, use_dask, 
    event_count_plot))

       
    def _start_callbacks(self,segimagedir, use_dask, 
    event_count_plot ):

           self._capture_csv_callback(segimagedir, use_dask)
           self._capture_plot_callback(segimagedir,event_count_plot)

    def update_start_prob(self, event):
        """update state of 'heatmapsteps' at current key-frame to reflect GUI state"""
        self.start_prob = self.frameWidget.startprobSpinBox.value()

    

    def update_nms_space(self, event):
        """update state of 'heatmapsteps' at current key-frame to reflect GUI state"""
        self.nms_space = self.frameWidget.nmsspaceSpinBox.value()    
 
    def update_heat_steps(self, event):
        """update state of 'heatmapsteps' at current key-frame to reflect GUI state"""
        self.heatmapsteps = self.frameWidget.heatstepsSpinBox.value()

    def updateLabel(self, value):

        real_value = float(self.start_prob + (1.0 - self.start_prob)/5000 * float(value)) 
        self.frameWidget.label.setText(str(real_value))
        self.event_threshold = float(real_value)

    def _capture_csv_callback(self, segimagedir, use_dask):
        
         get_image_text = self.frameWidget.imageidbox.currentText()
         csv_event_name = self.frameWidget.eventidbox.currentText()
         imagename = os.path.basename(os.path.splitext(get_image_text)[0])
         self.oneatvisualization.show_csv(imagename, csv_event_name, segimagedir = segimagedir, event_threshold = self.event_threshold, 
         use_dask = use_dask, heatmapsteps = self.heatmapsteps, nms_space = self.nms_space)
                 

    def _capture_image_callback(self, segimagedir, heatmapimagedir, heatname, use_dask):

         get_image_text = self.frameWidget.imageidbox.currentText()
         imagename = os.path.basename(os.path.splitext(get_image_text)[0])
         self.oneatvisualization.show_image(get_image_text, imagename, segimagedir, heatmapimagedir, heatname,
          use_dask)
   
    def _capture_plot_callback(self, segimagedir,event_count_plot):

         plot_event_name = self.frameWidget.plotidbox.currentText()
         self.oneatvisualization.show_plot( plot_event_name,event_count_plot,
           segimagedir, self.event_threshold)
