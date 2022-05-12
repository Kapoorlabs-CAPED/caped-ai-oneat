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
        imagename = None,
        csv_event_name = None,
        plot_event_name = None, 
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
        
        self.frameWidget.eventidbox.currentIndexChanged.connect(self._capture_csv_callback)
        self.default_reader = default_reader
        self.key_categories = key_categories
        # Create animation
        
        self.oneatvisualization = OneatVisualization(viewer, image_to_read, imagename, csv_event_name, plot_event_name, savedir, savename)
        # establish key bindings
        self._add_callbacks()

    def _add_callbacks(self):
        """Bind keys"""

        self.oneatvisualization.viewer.bind_key(
            "Ctrl-c", self._capture_csv_callback, overwrite=True
        )
       

       

    def _release_callbacks(self):
        """Release keys"""

        self.oneatvisualization.viewer.bind_key("Ctrl-c", None)

   

    def _set_current_frame(self):
        return self.frameWidget.startframeSpinBox.setValue(self.oneatvisualization.frame)

    def _capture_csv_callback(self, event=None):
         """Record current key-frame"""
         get_image_text = self.frameWidget.imageidbox.currentText()
         image = self.oneatvisualization.default_reader(get_image_text)
         csv_event_name = self.frameWidget.eventidbox.currentText()
         imagename = os.path.basename(os.path.splitext(get_image_text)[0])
         self.frameWidget.endframeSpinBox.setRange(0, self.image.shape[0])
         self.frameWidget.startframeSpinBox.setRange(0, self.image.shape[0])
         self.frameWidget.endframeSpinBox.setValue(self.image.shape[0])
         self.pathText.setText(self.savedir + imagename + '.gif')  
                  

    def _replace_keyframe_callback(self, event=None):
        """Replace current key-frame with new view"""
        self.animation.capture_keyframe(
            steps=self._get_interpolation_steps(),
            ease=self._get_easing_function(),
            insert=False,
        )
        self._set_current_frame()

    def _delete_keyframe_callback(self, event=None):
        """Delete current key-frame"""

        self.animation.key_frames.pop(self.animation.frame)
        self.animation.frame = (self.animation.frame - 1) % len(
            self.animation.key_frames
        )
        self.animation.set_to_keyframe(self.animation.frame)
        self._set_current_frame()

    def _key_adv_frame(self, event=None):
        """Go forwards in key-frame list"""

        new_frame = (self.animation.frame + 1) % len(self.animation.key_frames)
        self.animation.set_to_keyframe(new_frame)
        self._set_current_frame()

    def _key_back_frame(self, event=None):
        """Go backwards in key-frame list"""

        new_frame = (self.animation.frame - 1) % len(self.animation.key_frames)
        self.animation.set_to_keyframe(new_frame)
        self._set_current_frame()

    def _save_callback(self, event=None):

        """Record current key-frame"""
        self._capture_keyframe_callback()

        path = self.pathText.text()
        self.animation.animate(path)

    def close(self):
        self._release_callbacks()
        super().close()
