from qtpy.QtWidgets import QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton
from .OneatFrameWidget import OneatFrameWidget
import os
import napari
from dask.array.image import imread as daskread
from tifffile import imread,  imwrite
from ..OneatVisualization import OneatVisualization


class OneatWidget(QWidget):
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
        savedir: None,
        savename: None,
     
        key_categories : None,
        use_dask = False,
        event_threshold = None,
        segimagedir = None,
        heatimagedir = None,
        heatname = '_Heat',
        heatmapsteps = 1,
        start_project_mid = 0, 
        end_project_mid = 1,
        event_count_plot = 'Plot selected event count',
        cell_count_plot = 'Plot cell count',
        event_norm_count_plot = 'Plot selected normalized event count',
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
        
        self._layout.addWidget(animation)
        self.oneatvisualization = OneatVisualization(viewer ,key_categories,  savedir, savename, self.frameWidget.ax, self.frameWidget.figure)

        self.frameWidget.plotidbox.addItem(event_count_plot)
        self.frameWidget.plotidbox.addItem(cell_count_plot)
        self.frameWidget.plotidbox.addItem(event_norm_count_plot)
        self.frameWidget.heatstepsSpinBox = heatmapsteps
        self.frameWidget.scoreSlider.valueChanged.connect(self._on_slider_moved)


        self.frameWidget.imageidbox.currentIndexChanged.connect(lambda eventid = self.frameWidget.imageidbox :
        self._capture_image_callback(segimagedir, event_threshold, heatname, start_project_mid, end_project_mid, use_dask ))

        self.frameWidget.eventidbox.currentIndexChanged.connect(lambda eventid = self.frameWidget.eventidbox :
        self._capture_csv_callback(segimagedir, event_threshold, use_dask, heatmapsteps ))
        
 
        self.frameWidget.plotidbox.currentIndexChanged.connect(lambda eventid = self.frameWidget.imageidbox :
        self._capture_plot_callback(segimagedir,event_count_plot, cell_count_plot, event_norm_count_plot, use_dask, event_threshold))
       
        
    
    def _on_slider_moved(self, event=None):
        frame_index = event
        self.event_threshold = frame_index
        print(self.event_threshold)

    def _capture_csv_callback(self, segimagedir, event_threshold, use_dask, heatmapsteps):
        
         get_image_text = self.frameWidget.imageidbox.currentText()
         csv_event_name = self.frameWidget.eventidbox.currentText()
         imagename = os.path.basename(os.path.splitext(get_image_text)[0])
         self.oneatvisualization.show_csv(imagename, csv_event_name, segimagedir = segimagedir, event_threshold = event_threshold, 
         use_dask = use_dask, heatmapsteps = heatmapsteps)
                 

    def _capture_image_callback(self, segimagedir, heatmapimagedir, heatname, start_project_mid, end_project_mid, use_dask):

         get_image_text = self.frameWidget.imageidbox.currentText()
         imagename = os.path.basename(os.path.splitext(get_image_text)[0])
         self.oneatvisualization.show_image(get_image_text, imagename, segimagedir, heatmapimagedir, heatname,
         start_project_mid, end_project_mid, use_dask)
   
    def _capture_plot_callback(self, segimagedir,event_count_plot, cell_count_plot, event_norm_count_plot, use_dask, event_threshold):

         get_image_text = self.frameWidget.imageidbox.currentText()
         imagename = os.path.basename(os.path.splitext(get_image_text)[0])
         event_name = self.frameWidget.eventidbox.currentText()
         plot_event_name = self.frameWidget.plotidbox.currentText()
         self.oneatvisualization.show_plot(imagename, plot_event_name,event_count_plot,event_norm_count_plot,cell_count_plot,
          event_name,  use_dask, segimagedir, event_threshold)
