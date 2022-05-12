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
        default_reader: imread,
        key_categories : None,
        image_to_read: None,
        imagename : None,
        csv_event_name : None,
        plot_event_name : None, 
        use_dask = False,
        event_threshold = None,
        segimagedir = None,
        heatmapsteps = None,
        parent=None,
    ):
        super().__init__(parent=parent)

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self._layout.addWidget(QLabel('Oneat Visualization Wizard', parent=self))

        self.frameWidget = OneatFrameWidget(parent=self)
        self._layout.addWidget(self.frameWidget)
        self._layout.addStretch(1)

        self.pathText = QLineEdit(parent=self)
        self.pathText.setText(savedir + savename + '.gif')
        self._layout.addWidget(self.pathText)

        self.saveButton = QPushButton('Save Results', parent=self)
        self.saveButton.clicked.connect(self._save_callback)
        self._layout.addWidget(self.saveButton)
        
        self.frameWidget.heatstepsSpinBox = heatmapsteps

        self.frameWidget.eventidbox.currentIndexChanged.connect(self._capture_csv_callback(segimagedir, event_threshold, use_dask,heatmapsteps ))
        self.frameWidget.plotidbox.currentIndexChanged.connect(self._capture_plot_callback)
        # Create animation
        
        self.oneatvisualization = OneatVisualization(viewer, image_to_read, imagename, csv_event_name, 
        plot_event_name,default_reader,key_categories,  savedir, savename, self.frameWidget.ax, use_dask)
       
      
       

    
    
    def _capture_csv_callback(self, segimagedir, event_threshold, use_dask):
        
         get_image_text = self.frameWidget.imageidbox.currentText()
         csv_event_name = self.frameWidget.eventidbox.currentText()
         imagename = os.path.basename(os.path.splitext(get_image_text)[0])

         self.oneatvisualization.csv_event_name = csv_event_name
         self.oneatvisualization.image_to_read = get_image_text
         self.oneatvisualization.imagename = imagename 
         
      
         self.frameWidget.endframeSpinBox.setRange(0, self.image.shape[0])
         self.frameWidget.startframeSpinBox.setRange(0, self.image.shape[0])
         self.frameWidget.endframeSpinBox.setValue(self.image.shape[0])
         self.pathText.setText(self.oneatvisualization.savedir + imagename + '.gif')  
         self.oneatvisualization.show_csv(segimagedir = segimagedir, event_threshold = event_threshold, use_dask = use_dask, heatmapsteps = heatmapsteps)
                 

   

