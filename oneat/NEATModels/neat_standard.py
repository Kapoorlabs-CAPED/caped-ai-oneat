
from .neat_goldstandard import NEATDynamic

class NEATSDynamic(NEATDynamic):
   

    def __init__(self, config, model_dir, model_name, catconfig=None, cordconfig=None):

                super().__init__(config = config, model_dir = model_dir, model_name = model_name, catconfig = catconfig, cordconfig = cordconfig)

    def predict_standard(self, imagename, marker_tree, savedir, n_tiles=(1, 1), overlap_percent=0.8,
                event_threshold=0.5, iou_threshold=0.1, fidelity = 5, downsamplefactor = 1, normalize = True, center_oneat = True):

        self.predict(imagename,savedir,n_tiles = n_tiles, overlap_percent = overlap_percent, event_threshold = event_threshold, iou_threshold = iou_threshold, 
        fidelity = fidelity, downsamplefactor = downsamplefactor,  marker_tree = marker_tree, 
         remove_markers = None, normalize = normalize, center_oneat = center_oneat )



