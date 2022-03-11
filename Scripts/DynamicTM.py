import sys
import os
from glob import glob
from oneat.NEATModels import NEATDynamic, nets
from oneat.NEATModels.config import dynamic_config
from oneat.NEATUtils import helpers
from oneat.NEATUtils.helpers import save_json, load_json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"



npz_directory = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/oneatnpz/'
npz_name = 'Cellsplitdetectorm4p6.npz'
npz_val_name = 'Cellsplitdetectorm4p6val.npz'

#Read and Write the h5 file, directory location and name
model_dir =  '/data/u934/service_imagerie/v_kapoor/CurieDeepLearningModels/WinnerOneatModels/'
model_name = 'Cellsplitdetectorm4p6.h5'

#Neural network parameters
division_categories_json = model_dir + 'Cellsplitcategories.json'
key_categories = load_json(division_categories_json)
division_cord_json = model_dir + 'Cellsplitcord.json'
key_cord = load_json(division_cord_json)

#For ORNET use residual = True and for OSNET use residual = False
residual = True
#NUmber of starting convolutional filters, is doubled down with increasing depth
startfilter = 32
#CNN network start layer, mid layers and lstm layer kernel size
start_kernel = 7
lstm_kernel = 3
mid_kernel = 3
#Network depth has to be 9n + 2, n= 3 or 4 is optimal for Notum dataset
depth = 29
#Size of the gradient descent length vector, start small and use callbacks to get smaller when reaching the minima
learning_rate = 1.0E-3
#For stochastic gradient decent, the batch size used for computing the gradients
batch_size = 4
# use softmax for single event per box, sigmoid for multi event per box
lstm_hidden_unit = 16
#Training epochs, longer the better with proper chosen learning rate
epochs = 250
nboxes = 1
#The inbuilt model stride which is equal to the nulber of times image was downsampled by the network
show = False
stage_number = 3
last_conv_factor = 4
size_tminus = 4
size_tplus = 6
imagex = 64
imagey = 64
yolo_v0 = False
yolo_v1 = True
yolo_v2 = False


config = dynamic_config(npz_directory =npz_directory, npz_name = npz_name, npz_val_name = npz_val_name, 
                         key_categories = key_categories, key_cord = key_cord, nboxes = nboxes, imagex = imagex,
                         imagey = imagey, size_tminus = size_tminus, size_tplus =size_tplus, epochs = epochs, yolo_v0 = yolo_v0, yolo_v1 = yolo_v1, yolo_v2 = yolo_v2,
                         residual = residual, depth = depth, start_kernel = start_kernel, mid_kernel = mid_kernel, stage_number = stage_number, last_conv_factor = last_conv_factor,
                         lstm_kernel = lstm_kernel, lstm_hidden_unit = lstm_hidden_unit, show = show,
                         startfilter = startfilter, batch_size = batch_size, model_name = model_name)

config_json = config.to_json()

print(config)
save_json(config_json, model_dir + os.path.splitext(model_name)[0] + '_Parameter.json')

Train = NEATDynamic(config, model_dir, model_name)

Train.loadData()

Train.TrainModel()
