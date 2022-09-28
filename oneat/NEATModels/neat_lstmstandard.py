
from .neat_goldstandard import NEATDynamic
import numpy as np
class NEATLDynamic(NEATDynamic):
   

    def __init__(self, config, model_dir, model_name, catconfig=None, cordconfig=None, pure_lstm = True):

                super().__init__(config = config, model_dir = model_dir, model_name = model_name, catconfig = catconfig, cordconfig = cordconfig, pure_lstm = pure_lstm)

    def predict_standard(self, imagename,  savedir, n_tiles=(1, 1), overlap_percent=0.8, dtype = np.uint8,
                event_threshold=0.5, event_confidence = 0.5, iou_threshold=0.1,  fidelity=1, downsamplefactor = 1, start_project_mid = 4, end_project_mid = 4,
                marker_tree = None, remove_markers = False, normalize = True, center_oneat = True, nms_function = 'iou'):

        self.predict(self, imagename,  savedir, n_tiles=n_tiles, overlap_percent=overlap_percent, dtype = dtype,
                event_threshold=event_threshold, event_confidence = event_confidence, iou_threshold=iou_threshold,  fidelity=fidelity, downsamplefactor = downsamplefactor, start_project_mid = start_project_mid, end_project_mid = end_project_mid,
                marker_tree = marker_tree, remove_markers = remove_markers, normalize = normalize, center_oneat = center_oneat, nms_function = nms_function )



