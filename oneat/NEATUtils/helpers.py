from __future__ import print_function, unicode_literals, absolute_import, division
# import matplotlib.pyplot as plt
import numpy as np
import collections
import json
import cv2
from scipy import spatial
from skimage import measure
from pathlib import Path
import math
import os
import csv
from csbdeep.utils import normalize
from tifffile import imwrite
from tqdm import tqdm
from skimage.util import invert as invertimage
from skimage.measure import label
from skimage.filters import sobel
from scipy.ndimage.measurements import find_objects
from skimage.morphology import  dilation, square, binary_dilation, disk
from skimage import morphology
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import watershed
"""
 @author: Varun Kapoor

"""

"""
This method is used to convert Marker image to a list containing the XY indices for all time points
"""

def DownsampleData(image, DownsampleFactor):
                    
                if DownsampleFactor!=1:  
                    #print('Downsampling Image in XY by', DownsampleFactor)
                    scale_percent = int(100/DownsampleFactor) # percent of original size
                    width = int(image.shape[2] * scale_percent / 100)
                    height = int(image.shape[1] * scale_percent / 100)
                    dim = (width, height)
                    smallimage = np.zeros([image.shape[0],  height,width])
                    for i in range(0, image.shape[0]):
                          # resize image
                          smallimage[i,:] = cv2.resize(image[i,:].astype('float32'), dim)         
         
                    return smallimage
                else:
                    
                    return image
def remove_big_objects(ar, max_size=6400, connectivity=1, in_place=False):
    out = ar.copy()
    ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    too_big = component_sizes > max_size
    too_big_mask = too_big[ccs]
    out[too_big_mask] = 0

    return out


def IntergerLabelGen(BinaryImage, Name, savedir):
    InputBinaryImage = BinaryImage.astype('uint8')
    IntegerImage = np.zeros([BinaryImage.shape[0], BinaryImage.shape[1], BinaryImage.shape[2]])
    for i in tqdm(range(0, InputBinaryImage.shape[0])):
        BinaryImageOriginal = InputBinaryImage[i, :]
        Orig = normalizeFloatZeroOne(BinaryImageOriginal)
        InvertedBinaryImage = invertimage(BinaryImageOriginal)
        BinaryImage = normalizeFloatZeroOne(InvertedBinaryImage)
        image = binary_dilation(BinaryImage)
        image = invertimage(image)
        labelclean = label(image)
        labelclean = remove_big_objects(labelclean, max_size=15000)
        AugmentedLabel = dilation(labelclean, selem=square(3))
        AugmentedLabel = np.multiply(AugmentedLabel, Orig)
        IntegerImage[i, :] = AugmentedLabel

    imwrite(savedir + Name + '.tif', IntegerImage.astype('uint16'))


def MarkerToCSV(MarkerImage):
    MarkerImage = MarkerImage.astype('uint16')
    MarkerList = []
    print('Obtaining co-ordinates of markers in all regions')
    for i in range(0, MarkerImage.shape[0]):
        waterproperties = measure.regionprops(MarkerImage, MarkerImage)
        indices = [prop.centroid for prop in waterproperties]
        MarkerList.append([i, indices[0], indices[1]])
    return MarkerList


def load_json(fpath):
    with open(fpath, 'r') as f:
        return json.load(f)


def save_json(data, fpath, **kwargs):
    with open(fpath, 'w') as f:
        f.write(json.dumps(data, **kwargs))

        ##CARE csbdeep modification of implemented function


def normalizeFloat(x, pmin=3, pmax=99.8, axis=None, eps=1e-20, dtype=np.float32):
    """Percentile based Normalization
    
    Normalize patches of image before feeding into the network
    
    Parameters
    ----------
    x : np array Image patch
    pmin : minimum percentile value for normalization
    pmax : maximum percentile value for normalization
    axis : axis along which the normalization has to be carried out
    eps : avoid dividing by zero
    dtype: type of numpy array, float 32 default
    """
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, eps=eps, dtype=dtype)


def normalizeFloatZeroOne(x, pmin=3, pmax=99.8, axis=None, eps=1e-20, dtype=np.float32):
    """Percentile based Normalization
    
    Normalize patches of image before feeding into the network
    
    Parameters
    ----------
    x : np array Image patch
    pmin : minimum percentile value for normalization
    pmax : maximum percentile value for normalization
    axis : axis along which the normalization has to be carried out
    eps : avoid dividing by zero
    dtype: type of numpy array, float 32 default
    """
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalizer(x, mi, ma, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, eps=1e-20, dtype=np.float32):
    """
    Number expression evaluation for normalization
    
    Parameters
    ----------
    x : np array of Image patch
    mi : minimum input percentile value
    ma : maximum input percentile value
    eps: avoid dividing by zero
    dtype: type of numpy array, float 32 defaut
    """

    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    x = (x - mi) / (ma - mi + eps)

    return x


def normalizer(x, mi, ma, eps=1e-20, dtype=np.float32):
    """
    Number expression evaluation for normalization
    
    Parameters
    ----------
    x : np array of Image patch
    mi : minimum input percentile value
    ma : maximum input percentile value
    eps: avoid dividing by zero
    dtype: type of numpy array, float 32 defaut
    """

    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    x = (x - mi) / (ma - mi + eps)

    x = normalizeZeroOne(x)
    return x


def normalizeZeroOne(x):
    x = x.astype('float32')

    minVal = np.min(x)
    maxVal = np.max(x)

    x = ((x - minVal) / (maxVal - minVal + 1.0e-20))

    return x


def load_training_data(directory, filename, axes=None, verbose=True):
    """ Load training data in .npz format.
    The data file is expected to have the keys 'data' and 'label'     
    """
    if directory is not None:
        npzdata = np.load(directory + filename)
    else:
        npzdata = np.load(filename)

    X = npzdata['data']
    Y = npzdata['label']
    Z = npzdata['label2']

    if axes is None:
        axes = npzdata['axes']
    axes = axes_check_and_normalize(axes)
    assert 'C' in axes
    n_images = X.shape[0]
    assert X.shape[0] == Y.shape[0]
    assert 0 < n_images <= X.shape[0]

    X, Y = X[:n_images], Y[:n_images]
    channel = axes_dict(axes)['C']

    X = move_channel_for_backend(X, channel=channel)

    axes = axes.replace('C', '')  # remove channel
    if backend_channels_last():
        axes = axes + 'C'
    else:
        axes = axes[:1] + 'C' + axes[1:]

    if verbose:
        ax = axes_dict(axes)
        n_train = len(X)
        image_size = tuple(X.shape[ax[a]] for a in 'TZYX' if a in axes)
        n_dim = len(image_size)
        n_channel_in = X.shape[ax['C']]

        print('number of  images:\t', n_train)

        print('image size (%dD):\t\t' % n_dim, image_size)
        print('axes:\t\t\t\t', axes)
        print('channels in / out:\t\t', n_channel_in)

    return (X, Y, Z), axes


def _raise(e):
    raise e


# https://docs.python.org/3/library/itertools.html#itertools-recipes
def consume(iterator):
    collections.deque(iterator, maxlen=0)


def axes_check_and_normalize(axes, length=None, disallowed=None, return_allowed=False):
    """
    S(ample), T(ime), C(hannel), Z, Y, X
    """
    allowed = 'STCZYX'
    axes = str(axes).upper()
    consume(
        a in allowed or _raise(ValueError("invalid axis '%s', must be one of %s." % (a, list(allowed)))) for a in axes)
    disallowed is None or consume(a not in disallowed or _raise(ValueError("disallowed axis '%s'." % a)) for a in axes)
    consume(axes.count(a) == 1 or _raise(ValueError("axis '%s' occurs more than once." % a)) for a in axes)
    length is None or len(axes) == length or _raise(ValueError('axes (%s) must be of length %d.' % (axes, length)))
    return (axes, allowed) if return_allowed else axes


def axes_dict(axes):
    """
    from axes string to dict
    """
    axes, allowed = axes_check_and_normalize(axes, return_allowed=True)
    return {a: None if axes.find(a) == -1 else axes.find(a) for a in allowed}
    # return collections.namedt      

def backend_channels_last():
    import keras.backend as K
    assert K.image_data_format() in ('channels_first', 'channels_last')
    return K.image_data_format() == 'channels_last'


def move_channel_for_backend(X, channel):
    if backend_channels_last():
        return np.moveaxis(X, channel, -1)
    else:
        return np.moveaxis(X, channel, 1)
def load_full_training_data(directory, filename, axes=None, verbose=True):
    """ Load training data in .npz format.
    The data file is expected to have the keys 'data' and 'label'     
    """

    if directory is not None:
        npzdata = np.load(directory + filename)
    else:
        npzdata = np.load(filename)

    X = npzdata['data']
    Y = npzdata['label']

    if axes is None:
        axes = npzdata['axes']
    axes = axes_check_and_normalize(axes)
    assert 'C' in axes
    n_images = X.shape[0]
    assert X.shape[0] == Y.shape[0]
    assert 0 < n_images <= X.shape[0]

    X, Y = X[:n_images], Y[:n_images]
    channel = axes_dict(axes)['C']

    X = move_channel_for_backend(X, channel=channel)

    axes = axes.replace('C', '')  # remove channel
    if backend_channels_last():
        axes = axes + 'C'
    else:
        axes = axes[:1] + 'C' + axes[1:]

    if verbose:
        ax = axes_dict(axes)
        n_train = len(X)
        image_size = tuple(X.shape[ax[a]] for a in 'TZYX' if a in axes)
        n_dim = len(image_size)
        n_channel_in = X.shape[ax['C']]

        print('number of  images:\t', n_train)

        print('image size (%dD):\t\t' % n_dim, image_size)
        print('axes:\t\t\t\t', axes)
        print('channels in / out:\t\t', n_channel_in)

    return (X, Y), axes




def time_pad(image, TimeFrames):
    time = image.shape[0]

    timeextend = time

    while timeextend % TimeFrames != 0:
        timeextend = timeextend + 1

    extendimage = np.zeros([timeextend, image.shape[1], image.shape[2]])

    extendimage[0:time, :, :] = image

    return extendimage


def chunk_list(image, patchshape, stride, pair):
    rowstart = pair[0]
    colstart = pair[1]

    endrow = rowstart + patchshape[0]
    endcol = colstart + patchshape[1]

    if endrow > image.shape[1]:
        endrow = image.shape[1]
    if endcol > image.shape[2]:
        endcol = image.shape[2]

    region = (slice(0, image.shape[0]), slice(rowstart, endrow),
              slice(colstart, endcol))

    # The actual pixels in that region.
    patch = image[region]

    return patch, rowstart, colstart


def DensityCounter(MarkerImage, TrainshapeX, TrainshapeY):
    AllDensity = {}

    for i in tqdm(range(0, MarkerImage.shape[0])):
        density = []
        location = []
        currentimage = MarkerImage[i, :].astype('uint16')
        waterproperties = measure.regionprops(currentimage, currentimage)
        indices = [prop.centroid for prop in waterproperties]

        for y, x in indices:

            crop_Xminus = x - int(TrainshapeX / 2)
            crop_Xplus = x + int(TrainshapeX / 2)
            crop_Yminus = y - int(TrainshapeY / 2)
            crop_Yplus = y + int(TrainshapeY / 2)

            region = (slice(int(crop_Yminus), int(crop_Yplus)),
                      slice(int(crop_Xminus), int(crop_Xplus)))
            crop_image = currentimage[region].astype('uint16')
            if crop_image.shape[0] >= TrainshapeY and crop_image.shape[1] >= TrainshapeX:
                waterproperties = measure.regionprops(crop_image, crop_image)

                labels = [prop.label for prop in waterproperties]
                labels = np.asarray(labels)
                # These regions should be downsampled
                density.append(labels.shape[0])
                location.append((int(y), int(x)))
        # Create a list of TYX marker locations that should be downsampled
        AllDensity[str(i)] = [density, location]

    return AllDensity





def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""

    # TODO: refactor 'fill_label_holes' and 'edt_prob' to share code
    def grow(sl, interior):
        return tuple(slice(s.start - int(w[0]), s.stop + int(w[1])) for s, w in zip(sl, interior))

    def shrink(interior):
        return tuple(slice(int(w[0]), (-1 if w[1] else None)) for w in interior)

    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i, sl in enumerate(objects, 1):
        if sl is None: continue
        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl, interior)] == i
        mask_filled = binary_fill_holes(grown_mask, **kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


def dilate_label_holes(lbl_img, iterations):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (range(np.min(lbl_img), np.max(lbl_img) + 1)):
        mask = lbl_img == l
        mask_filled = binary_dilation(mask, iterations=iterations)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled

def zero_pad(image, TrainshapeX, TrainshapeY):
    time = image.shape[0]
    sizeY = image.shape[2]
    sizeX = image.shape[1]

    sizeXextend = sizeX
    sizeYextend = sizeY

    while sizeXextend % TrainshapeX != 0:
        sizeXextend = sizeXextend + 1

    while sizeYextend % TrainshapeY != 0:
        sizeYextend = sizeYextend + 1

    extendimage = np.zeros([time, sizeXextend, sizeYextend])

    extendimage[0:time, 0:sizeX, 0:sizeY] = image

    return extendimage


def twod_zero_pad(image, PadX, PadY):
    sizeY = image.shape[1]
    sizeX = image.shape[0]

    sizeXextend = sizeX
    sizeYextend = sizeY

    while sizeXextend % PadX != 0:
        sizeXextend = sizeXextend + 1

    while sizeYextend % PadY != 0:
        sizeYextend = sizeYextend + 1

    extendimage = np.zeros([sizeXextend, sizeYextend])

    extendimage[0:sizeX, 0:sizeY] = image

    return extendimage

def Generate_only_mask(Image, maskmodel, n_tiles):
  Mask = np.zeros([Image.shape[0], Image.shape[1], Image.shape[2]])
  for i in (range(0, Image.shape[0])):
        smallimage = Image[i, :]
        maskimage = GenerateMask(smallimage, maskmodel, n_tiles)
        maskimage = fill_label_holes(maskimage.astype('uint16'))

        Mask[i,:] = maskimage
  return Mask


def MidSlices(Image, start_project_mid, end_project_mid, axis = 1):
    
    
    SmallImage = Image.take(indices = range(Image.shape[axis]//2 - start_project_mid, Image.shape[axis]//2 + end_project_mid), axis = axis)
    
    MaxProject = np.amax(SmallImage, axis = axis)
        
    return MaxProject


def GenerateMarkers(Image, n_tiles, model = None,  maskmodel = None, segimage = None, 
            start_project_mid = 4, end_project_mid = 4):

    ndim = len(Image.shape)
    Markers = np.zeros([Image.shape[0], Image.shape[-2], Image.shape[-1]])
    if segimage is None:
        Watershed = np.zeros([Image.shape[0], Image.shape[-2], Image.shape[-1]])
        Star = np.zeros([Image.shape[0], Image.shape[-2], Image.shape[-1]])
    else:
        Watershed = segimage
        Star = segimage
        if ndim == 4:
                    Watershed =  MidSlices(Watershed, start_project_mid, end_project_mid, axis = 1)
                    Star =  MidSlices(Star, start_project_mid, end_project_mid, axis = 1)
    if maskmodel is not None:
        Mask = np.zeros([Image.shape[0], Image.shape[-2], Image.shape[-1]])
    else:
        Mask = None  
           

    if segimage is None and model is not None:
            
            for i in tqdm(range(0, Image.shape[0])):
                smallimage = Image[i, :]
                if maskmodel is not None:
                        maskimage = GenerateMask(smallimage, maskmodel, n_tiles)
                        maskimage = fill_label_holes(maskimage.astype('uint16'))
                        if ndim == 4:
                            maskimage =  MidSlices(maskimage, start_project_mid, end_project_mid, axis = 0)
                        Mask[i,:] = maskimage
                else:
                        maskimage = None
                if ndim == 3:                
                   smallimage = normalize(smallimage, 1, 99.8, axis=(0, 1))
                if ndim == 4:
                   smallimage = normalize(smallimage, 1, 99.8, axis=(0, 1, 2)) 
                starimage, details = model.predict_instances(smallimage, n_tiles=n_tiles)
                properties = measure.regionprops(starimage)
                Coordinates = [prop.centroid for prop in properties]
                if ndim == 3:
                    Coordinates = sorted(Coordinates, key=lambda k: [k[1], k[0]])
                    Coordinates.append((0, 0))
                if ndim == 4:
                    Coordinates = sorted(Coordinates, key=lambda k: [k[2], k[1], k[0]])
                    Coordinates.append((0, 0, 0))    
                Coordinates = np.asarray(Coordinates)

                coordinates_int = np.round(Coordinates).astype(int)
                markers_raw = np.zeros_like(smallimage)
                markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))

                markers = dilation(markers_raw, disk(2))
                if ndim == 4:
                            markers = morphology.dilation(
                               markers_raw.astype('uint16'), morphology.ball(2))
                if ndim == 3:       
                            markers = morphology.dilation(markers_raw.astype('uint16'), morphology.disk(2))
                watershedImage = watershed(-smallimage, markers, mask=maskimage.copy())
                if ndim == 4:
                            markers =  MidSlices(markers, start_project_mid, end_project_mid, axis = 0)
                            watershedImage =  MidSlices(watershedImage, start_project_mid, end_project_mid, axis = 0)
                            starimage =  MidSlices(starimage, start_project_mid, end_project_mid, axis = 0)
                Markers[i, :] = label(markers.astype('uint16'))
                Watershed[i,:] = watershedImage
                Star[i,:] = starimage
    if segimage is not None:
          
       
                for i in tqdm(range(0, segimage.shape[0])):
                        properties = measure.regionprops(segimage[i,:])
                        smallimage = Image[i, :]
                        if maskmodel is not None:
                                maskimage = GenerateMask(smallimage, maskmodel, n_tiles)
                                maskimage = fill_label_holes(maskimage.astype('uint16'))
                                if ndim == 4:
                                     maskimage =  MidSlices(maskimage, start_project_mid, end_project_mid, axis = 0)

                                Mask[i,:] = maskimage
                        else:
                                maskimage = None
                        Coordinates = [prop.centroid for prop in properties]
 
                        if ndim == 3:
                            Coordinates = sorted(Coordinates, key=lambda k: [k[1], k[0]])
                            Coordinates.append((0, 0))
                        if ndim == 4:
                            Coordinates = sorted(Coordinates, key=lambda k: [k[2], k[1], k[0]])
                            Coordinates.append((0, 0, 0))
                        Coordinates = np.asarray(Coordinates)
    
                        coordinates_int = np.round(Coordinates).astype(int)
                        markers_raw = np.zeros_like(smallimage)
                        markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))
                        if ndim == 4:
                            markers = morphology.dilation(
                               markers_raw.astype('uint16'), morphology.ball(2))
                        if ndim == 3:       
                            markers = morphology.dilation(markers_raw.astype('uint16'), morphology.disk(2)) 
                        if ndim == 4:
                            markers =  MidSlices(markers, start_project_mid, end_project_mid, axis = 0)

                        Markers[i, :] = label(markers.astype('uint16'))


    return Markers, Watershed, Star, Mask


def GenerateMask(Image, model, n_tiles):
     
    Segmented = model.predict(Image, 'YX', n_tiles=n_tiles)
    try:
            thresholds = threshold_multiotsu(Segmented, classes=2)

            # Using the threshold values, we generate the three regions.
            regions = np.digitize(Segmented, bins=thresholds)
    except:

            regions = Segmented

    Binary = regions > 0
        

    return Binary  

"""
This method takes the integer labelled segmentation image as input and creates a dictionary of markers at all timepoints for easy search
"""


def MakeTrees(segimage):
    AllTrees = {}
    print("Creating Dictionary of marker location for fast search")
    for i in tqdm(range(0, segimage.shape[0])):

        indices = []
        
        currentimage = segimage[i, :].astype('uint16')
       
        currentimage = currentimage > 0
        currentimage = label(currentimage)
        props = measure.regionprops(currentimage)
        for prop in props:
              indices.append((int(prop.centroid[0]), int(prop.centroid[1])))
        # Comparison between image_max and im to find the coordinates of local maxima
      
        if len(indices) > 0:
            tree = spatial.cKDTree(indices)

            AllTrees[str(i)] = [tree, indices]

    return AllTrees


       


def get_max_score_index(scores, threshold=0, top_k=0, descending=True):
    """ Get the max scores with corresponding indicies
    Adapted from the OpenCV c++ source in `nms.inl.hpp <https://github.com/opencv/opencv/blob/ee1e1ce377aa61ddea47a6c2114f99951153bb4f/modules/dnn/src/nms.inl.hpp#L33>`__
    :param scores: a list of scores
    :type scores: list
    :param threshold: consider scores higher than this threshold
    :type threshold: float
    :param top_k: return at most top_k scores; if 0, keep all
    :type top_k: int
    :param descending: if True, list is returened in descending order, else ascending
    :returns: a  sorted by score list  of [score, index]
    """
    score_index = []

    # Generate index score pairs
    for i, score in enumerate(scores):
        if (threshold > 0) and (score > threshold):
            score_index.append([score, i])
        else:
            score_index.append([score, i])

    # Sort the score pair according to the scores in descending order
    npscores = np.array(score_index)

    if descending:
        npscores = npscores[npscores[:, 0].argsort()[::-1]]  # descending order
    else:
        npscores = npscores[npscores[:, 0].argsort()]  # ascending order

    if top_k > 0:
        npscores = npscores[0:top_k]

    return npscores.tolist()

              


def compare_function_sec(box1, box2, gridx, gridy):
    w1, h1 = box1['width'], box1['height']
    w2, h2 = box2['width'], box2['height']
    x1 = box1['xstart']
    x2 = box2['xstart']
    
    y1 = box1['ystart']
    y2 = box2['ystart']

    if abs(x1 - x2) <= gridx and abs(y1 - y2) <= gridy:
           
            xA = max(x1 , x2 )
            xB = min(x1 + w1, x2 + w2)
            yA = max(y1, y2)
            yB = min(y1 + h1, y2+ h2)

            intersect = max(0, xB - xA ) * max(0, yB - yA )

            area = h2 * w2 + h1 * w1 - intersect

            return float(np.true_divide(intersect, area))
    else:

        return -2 






def goodboxes(boxes, scores, nms_threshold, score_threshold, gridx, gridy,
               fidelity=1):



    if len(boxes) == 0:
        return []

    assert len(scores) == len(boxes)
    assert scores is not None
    if scores is not None:
        assert len(scores) == len(boxes)

    boxes = np.array(boxes)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")




    # initialize the list of picked indexes
    pick = []
    Averageboxes = []
    # sort the bounding boxes by the associated scores
    scores = get_max_score_index(scores, score_threshold, 0, False)
    idxs = np.array(scores, np.int32)[:, 1]

    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        count = 0
        # loop over all indexes in the indexes list
        for pos in (range(0, last)):
            # grab the current index
                j = idxs[pos]

                overlap = compare_function_sec(boxes[i], boxes[j], gridx, gridy)
                # if there is sufficient overlap, suppress the current bounding box
                if overlap > abs(nms_threshold):
                        count = count + 1
                        if count >= fidelity:
                            
                            if boxes[i] not in Averageboxes:
                              Averageboxes.append(boxes[i])
                        suppress.append(pos)
            # delete all indexes from the index list that are in the suppression list
        idxs = np.delete(idxs, suppress)
    # return only the indicies of the bounding boxes that were picked
    return Averageboxes



def simpleaveragenms(boxes, scores, nms_threshold, score_threshold, event_name, gridx, gridy):
    if len(boxes) == 0:
        return []

    assert len(scores) == len(boxes)
    assert scores is not None
    if scores is not None:
        assert len(scores) == len(boxes)

    boxes = np.array(boxes)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []
    Averageboxes = []
    newbox = None
    # sort the bounding boxes by the associated scores
    scores = get_max_score_index(scores, score_threshold, 0, False)
    idxs = np.array(scores, np.int32)[:, 1]

    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        count = 0
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # compute the ratio of overlap between the two boxes and the area of the second box
            overlap = compare_function_sec(boxes[i], boxes[j], gridx, gridy)

            # if there is sufficient overlap, suppress the current bounding box
            if overlap > nms_threshold:
                boxAscore = boxes[i][event_name]
                boxAXstart = boxes[i]['xstart']
                boxAYstart = boxes[i]['ystart']
                boxAXcenter = boxes[i]['xcenter']
                boxAYcenter = boxes[i]['ycenter']
                boxArealz = boxes[i]['real_z_event']
                boxAheight = boxes[i]['height']
                boxAwidth = boxes[i]['width']
                boxAconfidence = boxes[i]['confidence']

                boxBscore = boxes[j][event_name]
                boxBXstart = boxes[j]['xstart']
                boxBYstart = boxes[j]['ystart']
                boxBXcenter = boxes[j]['xcenter']
                boxBYcenter = boxes[j]['ycenter']
                boxBrealz = boxes[j]['real_z_event']
                boxBheight = boxes[j]['height']
                boxBwidth = boxes[j]['width']
                boxBconfidence = boxes[j]['confidence']

                meanboxscore = (boxAscore + boxBscore) / 2
                meanboxXstart = (boxAXstart + boxBXstart) / 2
                meanboxYstart = (boxAYstart + boxBYstart) / 2
                meanboxXcenter = (boxAXcenter + boxBXcenter) / 2
                meanboxYcenter = (boxAYcenter + boxBYcenter) / 2
                meanboxrealz = (boxArealz + boxBrealz) / 2
                meanboxheight = (boxAheight + boxBheight) / 2
                meanboxwidth = (boxAwidth + boxBwidth) / 2
                meanboxconfidence = (boxAconfidence + boxBconfidence) / 2

                newbox = {'xstart': meanboxXstart, 'ystart': meanboxYstart, 'xcenter': meanboxXcenter,
                          'ycenter': meanboxYcenter, 'real_z_event': meanboxrealz,
                          'height': meanboxheight, 'width': meanboxwidth, 'confidence': meanboxconfidence,
                          event_name: meanboxscore}
                suppress.append(pos) 
               
           
                

        if newbox is not None and newbox not in Averageboxes:
            Averageboxes.append(newbox)
            # delete all indexes from the index list that are in the suppression list
        idxs = np.delete(idxs, suppress)
    # return only the indicies of the bounding boxes that were picked

    meanscore = sum(d[event_name] for d in Averageboxes) / max(1,len(Averageboxes))
    maxscore = max(d[event_name] for d in Averageboxes)

    box = {'real_z_event': meanboxrealz, 'confidence': meanboxconfidence, event_name: meanscore, 'max_score': maxscore}

    return box, Averageboxes



def save_static_csv(ColorimageStatic,ColorimageDynamic,imagename, key_categories, iou_classedboxes, savedir, downsamplefactor, maskimage = None):
    
    
    for (event_name, event_label) in key_categories.items():

            if event_label > 0:
                xlocations = []
                ylocations = []
                scores = []
                confidences = []
                tlocations = []
                radiuses = []

                iou_current_event_boxes = iou_classedboxes[event_name][0]
                iou_current_event_boxes = sorted(iou_current_event_boxes, key=lambda x: x[event_name], reverse=True)

                for iou_current_event_box in iou_current_event_boxes:
                    xcenter = iou_current_event_box['xcenter'] * downsamplefactor
                    ycenter = iou_current_event_box['ycenter'] * downsamplefactor
                    tcenter = iou_current_event_box['real_time_event']
                    confidence = iou_current_event_box['confidence']
                    score = iou_current_event_box[event_name]
                    radius = np.sqrt(
                        iou_current_event_box['height'] * iou_current_event_box['height'] + iou_current_event_box[
                            'width'] * iou_current_event_box['width']) // 2
                    radius = radius * downsamplefactor
                    # Replace the detection with the nearest marker location
                    if maskimage is not None:
                        if maskimage[int(tcenter), int(ycenter), int(xcenter)] > 0:

                            xlocations.append(xcenter)
                            ylocations.append(ycenter)
                            scores.append(score)
                            confidences.append(confidence)
                            tlocations.append(tcenter)
                            radiuses.append(radius)
                    else:
                            xlocations.append(xcenter)
                            ylocations.append(ycenter)
                            scores.append(score)
                            confidences.append(confidence)
                            tlocations.append(tcenter)
                            radiuses.append(radius)


                event_count = np.column_stack([tlocations, ylocations, xlocations, scores, radiuses, confidences])
                event_count = sorted(event_count, key=lambda x: x[0], reverse=False)
                event_data = []
                csvname = savedir + "/" + event_name + "Location" + (
                os.path.splitext(os.path.basename(imagename))[0])
                writer = csv.writer(open(csvname + ".csv", "a"))
                filesize = os.stat(csvname + ".csv").st_size
                if filesize < 1:
                    writer.writerow(['T', 'Y', 'X', 'Score', 'Size', 'Confidence'])
                for line in event_count:
                    if line not in event_data:
                        event_data.append(line)
                    writer.writerows(event_data)
                    event_data = []
                    
                saveimage(ColorimageStatic,ColorimageDynamic , xlocations, ylocations, tlocations, scores, radius, event_label)    
                    
def saveimage(ColorimageStatic,ColorimageDynamic , xlocations, ylocations, tlocations, scores, radius, event_label):

        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
        # fontScale
        fontScale = 1

        # Blue color in BGR
        textcolor = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2
        for j in range(len(xlocations)):
            startlocation = (int(xlocations[j] - radius[j]), int(ylocations[j] - radius[j]))
            endlocation = (int(xlocations[j] + radius[j]), int(ylocations[j] + radius[j]))
            Z = int(tlocations[j])
            if Z < ColorimageDynamic.shape[0] - 1:
                    if event_label == 1:
                        image = ColorimageDynamic[Z, :, :, 1]
                        color = (0, 255, 0)
                    if event_label == 2:
                        color = (0, 0, 255)
                        image = ColorimageDynamic[Z, :, :, 2]
                    if event_label == 3:
                        color = (0, 255, 0)
                        image = ColorimageStatic[Z, :, :, 0]
                    if event_label == 4:
                        color = (0, 0, 255)
                        image = ColorimageStatic[Z, :, :, 1]
                    if event_label == 5:
                        color = (255, 0, 0)
                        image = ColorimageStatic[Z, :, :, 2]

                    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    cv2.rectangle(img, startlocation, endlocation, textcolor, thickness)
        
                    cv2.putText(img, str('%.5f' % (scores[j])), startlocation, cv2.FONT_HERSHEY_SIMPLEX, 1, textcolor,
                                thickness, cv2.LINE_AA)
                    if event_label == 1:
                        ColorimageDynamic[Z, :, :, 1] = img[:, :, 0]
                    if event_label == 2:
                        ColorimageDynamic[Z, :, :, 2] = img[:, :, 0]
                    if event_label == 3:
                        ColorimageStatic[Z, :, :, 0] = img[:, :, 0]
                    if event_label == 4:
                        ColorimageStatic[Z, :, :, 1] = img[:, :, 0]
                    if event_label == 5:
                        ColorimageStatic[Z, :, :, 2] = img[:, :, 0]

def gold_nms(heatmap, classedboxes, event_name, downsamplefactor, iou_threshold, event_threshold, gridx, gridy, fidelity):

               sorted_event_box = classedboxes[event_name][0]
               scores = [ sorted_event_box[i][event_name]  for i in range(len(sorted_event_box))]
               
               good_sorted_event_box = goodboxes(sorted_event_box, scores, iou_threshold, event_threshold,  gridx, gridy, fidelity)
               filtered_good_sorted_event_box = []
               for iou_current_event_box in good_sorted_event_box:
                                                              xcenter = iou_current_event_box['xcenter']* downsamplefactor
                                                              ycenter = iou_current_event_box['ycenter']* downsamplefactor
                                                              tcenter = iou_current_event_box['real_time_event']
                                                              score = iou_current_event_box[event_name]
                                                              filtered_good_sorted_event_box.append(iou_current_event_box)
                                                              for x in range(int(xcenter - 8), int(xcenter + 8)):
                                                                  for y in range(int(ycenter - 8), int(ycenter + 8)):

                                                                      heatmap[int(tcenter), int(y), int(x)] = heatmap[int(tcenter), int(y), int(x)] + score

               scores = [ filtered_good_sorted_event_box[i][event_name]  for i in range(len(filtered_good_sorted_event_box))]
                    
               return filtered_good_sorted_event_box






def dynamic_nms(heatmap, maskimage, classedboxes, event_name, downsamplefactor, iou_threshold, event_threshold, gridx, gridy, fidelity):
                
               sorted_event_box = classedboxes[event_name][0]
               scores = [ sorted_event_box[i][event_name]  for i in range(len(sorted_event_box))]
               best_sorted_event_box = goodboxes(sorted_event_box, scores, iou_threshold, event_threshold,  gridx, gridy, fidelity)
               
               filtered_good_sorted_event_box = []
               for iou_current_event_box in best_sorted_event_box:
                                                          xcenter = iou_current_event_box['xcenter']* downsamplefactor
                                                          ycenter = iou_current_event_box['ycenter']* downsamplefactor
                                                          tcenter = iou_current_event_box['real_time_event']
                                                          score = iou_current_event_box[event_name]
                                                      
                                                          if maskimage is not None:
                                                              if maskimage[int(tcenter), int(ycenter), int(xcenter)] > 0:
                                                                      filtered_good_sorted_event_box.append(iou_current_event_box)
                                                                      for x in range(int(xcenter - 8), int(xcenter + 8)):
                                                                          for y in range(int(ycenter - 8), int(ycenter + 8)):
                                                                              if x < heatmap.shape[2] and y < heatmap.shape[1]:
                                                                                  heatmap[int(tcenter), int(y), int(x)] = heatmap[int(tcenter), int(y), int(x)] + score
                                                                              
                                                          else:
                                                              
                                                              filtered_good_sorted_event_box.append(iou_current_event_box)
                                                              for x in range(int(xcenter - 8), int(xcenter + 8)):
                                                                  for y in range(int(ycenter - 8), int(ycenter + 8)):
                                                                      
                                                                      heatmap[int(tcenter), int(y), int(x)] = heatmap[int(tcenter), int(y), int(x)] + score
                                                                      
               
               return best_sorted_event_box
           

def microscope_dynamic_nms( classedboxes, event_name, iou_threshold, event_threshold, gridx, gridy, fidelity):
    
               sorted_event_box = classedboxes[event_name][0]
               scores = [ sorted_event_box[i][event_name]  for i in range(len(sorted_event_box))]
               best_sorted_event_box = goodboxes(sorted_event_box, scores, iou_threshold, event_threshold,  gridx, gridy, fidelity)
               
               return best_sorted_event_box

def save_dynamic_csv(imagename, key_categories, iou_classedboxes, savedir, downsamplefactor, ndim, z = 0, maskimage = None):
    
        for (event_name, event_label) in key_categories.items():

            if event_label > 0:
                xlocations = []
                ylocations = []
                zlocations = []
                scores = []
                confidences = []
                tlocations = []
                radiuses = []
                angles = []

                iou_current_event_boxes = iou_classedboxes[event_name][0]
                iou_current_event_boxes = sorted(iou_current_event_boxes, key=lambda x: x[event_name], reverse=True)
                for iou_current_event_box in iou_current_event_boxes:
                    xcenter = iou_current_event_box['xcenter'] * downsamplefactor
                    ycenter = iou_current_event_box['ycenter'] * downsamplefactor
                    tcenter = iou_current_event_box['real_time_event']
                    confidence = iou_current_event_box['confidence']
                    angle = iou_current_event_box['realangle']
                    score = iou_current_event_box[event_name]
                    radius = np.sqrt(
                        iou_current_event_box['height'] * iou_current_event_box['height'] + iou_current_event_box[
                            'width'] * iou_current_event_box['width']) // 2
                    radius = radius * downsamplefactor

                    if maskimage is not None:
                        if maskimage[int(tcenter), int(ycenter), int(xcenter)] > 0:
                                xlocations.append(xcenter)
                                ylocations.append(ycenter)
                                scores.append(score)
                                confidences.append(confidence)
                                tlocations.append(tcenter)
                                radiuses.append(radius)
                                angles.append(angle)
                                zlocations.append(z)
                    else:
                                xlocations.append(xcenter)
                                ylocations.append(ycenter)
                                scores.append(score)
                                confidences.append(confidence)
                                tlocations.append(tcenter)
                                radiuses.append(radius)
                                angles.append(angle)
                                zlocations.append(z)        
                event_count = np.column_stack(
                            [tlocations, zlocations, ylocations, xlocations, scores, radiuses, confidences, angles])
                event_count = sorted(event_count, key=lambda x: x[0], reverse=False)
                event_data = []
                csvname = savedir + "/" + event_name + "Location" + (
                os.path.splitext(os.path.basename(imagename))[0])
                writer = csv.writer(open(csvname + ".csv", "a"))
                filesize = os.stat(csvname + ".csv").st_size

                if filesize < 1:
                            writer.writerow(['T', 'Z', 'Y', 'X', 'Score', 'Size', 'Confidence', 'Angle'])
                for line in event_count:
                            if line not in event_data:
                                event_data.append(line)
                            writer.writerows(event_data)
                            event_data = []
    
    

def area_function(boxes):
    """Calculate the area of each polygon in polys

    :param polys: a list of polygons, each specified by its verticies
    :type polys: list
    :return: a list of areas corresponding the list of polygons
    :rtype: list
    """
    areas = []
    for poly in boxes:
        areas.append(cv2.contourArea(np.array(poly, np.int32)))
    return areas


def get_max_score_index(scores, threshold=0, top_k=0, descending=True):
    """ Get the max scores with corresponding indicies

    Adapted from the OpenCV c++ source in `nms.inl.hpp <https://github.com/opencv/opencv/blob/ee1e1ce377aa61ddea47a6c2114f99951153bb4f/modules/dnn/src/nms.inl.hpp#L33>`__

    :param scores: a list of scores
    :type scores: list
    :param threshold: consider scores higher than this threshold
    :type threshold: float
    :param top_k: return at most top_k scores; if 0, keep all
    :type top_k: int
    :param descending: if True, list is returened in descending order, else ascending
    :returns: a  sorted by score list  of [score, index]
    """
    score_index = []

    # Generate index score pairs
    for i, score in enumerate(scores):
        if (threshold > 0) and (score > threshold):
            score_index.append([score, i])
        else:
            score_index.append([score, i])

    # Sort the score pair according to the scores in descending order
    npscores = np.array(score_index)

    if descending:
        npscores = npscores[npscores[:, 0].argsort()[::-1]]  # descending order
    else:
        npscores = npscores[npscores[:, 0].argsort()]  # ascending order

    if top_k > 0:
        npscores = npscores[0:top_k]

    return npscores.tolist()


"""
This method is used to create a segmentation image of an input image (StarDist probability or distance map) using marker controlled watershedding using a mask image (UNET) 
"""


def WatershedwithMask(Image, Label, mask, grid):
    properties = measure.regionprops(Label, Image)
    Coordinates = [prop.centroid for prop in properties]
    Coordinates = sorted(Coordinates, key=lambda k: [k[1], k[0]])
    Coordinates.append((0, 0))
    Coordinates = np.asarray(Coordinates)

    coordinates_int = np.round(Coordinates).astype(int)
    markers_raw = np.zeros_like(Image)
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))

    markers = dilation(markers_raw, disk(2))
    Image = sobel(Image)
    watershedImage = watershed(Image, markers, mask=mask)

    return watershedImage, markers


"""
Prediction function for whole image/tile, output is Prediction vector for each image patch it passes over
"""


def yoloprediction(sy, sx, time_prediction, stride, inputtime, config, key_categories, key_cord, nboxes, mode,
                   event_type, marker_tree=None):
    LocationBoxes = []
    j = 0
    k = 1
    while True:
        j = j + 1
        if j > time_prediction.shape[1]:
            j = 1
            k = k + 1

        if k > time_prediction.shape[0]:
            break;
        Classybox = predictionloop(j, k, sx, sy, nboxes, stride, time_prediction, config, key_categories, key_cord,
                                   inputtime, mode, event_type, marker_tree = marker_tree)
        # Append the box and the maximum likelehood detected class
        if Classybox is not None:
                LocationBoxes.append(Classybox)

    return LocationBoxes


def focyoloprediction(sy, sx, z_prediction, stride, inputz, config, key_categories, key_cord, nboxes):
    LocationBoxes = []
    j = 0
    k = 1
    while True:
        j = j + 1
        if j > z_prediction.shape[2]:
            j = 1
            k = k + 1

        if k > z_prediction.shape[1]:
            break;

        Classybox = focpredictionloop(j, k, sx, sy, nboxes, stride, z_prediction, config, key_categories, key_cord,
                                      inputz)
        # Append the box and the maximum likelehood detected class
        if Classybox is not None:
            LocationBoxes.append(Classybox)
    return LocationBoxes


def nonfcn_yoloprediction(sy, sx, time_prediction, stride, inputtime, config, key_categories, key_cord, nboxes, mode,
                          event_type, marker_tree=None):
    LocationBoxes = []
    j = 1
    k = 1
    Classybox = predictionloop(j, k, sx, sy, nboxes, stride, time_prediction, config, key_categories, key_cord,
                               inputtime, mode, event_type, marker_tree = marker_tree)
    # Append the box and the maximum likelehood detected class
    if Classybox is not None:
        LocationBoxes.append(Classybox)
    return LocationBoxes


def predictionloop(j, k, sx, sy, nboxes, stride, time_prediction, config, key_categories, key_cord, inputtime, mode,
                   event_type, marker_tree = None):
    total_classes = len(key_categories)
    total_coords = len(key_cord)
    y = (k - 1) * stride
    x = (j - 1) * stride
   
    prediction_vector = time_prediction[k - 1, j - 1, :]

    xstart = x + sx
    ystart = y + sy
    Class = {}
    # Compute the probability of each class
    for (event_name, event_label) in key_categories.items():
        Class[event_name] = prediction_vector[event_label]

    xcentermean = 0
    ycentermean = 0
    tcentermean = 0
    xcenterrawmean = 0
    ycenterrawmean = 0
    tcenterrawmean = 0
    boxtcentermean = 0
    widthmean = 0
    heightmean = 0
    anglemean = 0
    boxtstartmean = 0
    angle = 0
    boxtstart = inputtime
    tcenter = 0
    boxtcenter = 0
    confidencemean = 0
    trainshapex = config['imagex']
    trainshapey = config['imagey']
    tcenterraw = 0
    for b in range(0, nboxes):
        xcenter = xstart + prediction_vector[total_classes + config['x'] + b * total_coords] * trainshapex
        ycenter = ystart + prediction_vector[total_classes + config['y'] + b * total_coords] * trainshapey
        xcenterraw = prediction_vector[total_classes + config['x'] + b * total_coords]
        ycenterraw = prediction_vector[total_classes + config['y'] + b * total_coords]

        try:
            height = prediction_vector[total_classes + config['h'] + b * total_coords] * trainshapex
            width = prediction_vector[total_classes + config['w'] + b * total_coords] * trainshapey
        except:
            height = trainshapey
            width = trainshapex
            pass
        if event_type == 'dynamic' and mode == 'detection':
            time_frames = config['size_tminus'] + config['size_tplus'] + 1
            tcenter = int(((inputtime + (prediction_vector[total_classes + config['t'] + b * total_coords] * time_frames))))
            tcenterraw = prediction_vector[total_classes + config['t'] + b * total_coords]
            boxtcenter = int((prediction_vector[total_classes + config['t'] + b * total_coords]))
            boxtstart = inputtime
            if config['yolo_v2']:
                angle = prediction_vector[total_classes + config['angle'] + b * total_coords]
                confidence = prediction_vector[total_classes + config['c'] + b * total_coords]
            if config['yolo_v1']:
                angle = 2
                confidence = prediction_vector[total_classes + config['c'] + b * total_coords]
            if config['yolo_v0']:
                angle = 2
                confidence = 1
        if mode == 'prediction':
            angle = 2
            confidence = 1

        if event_type == 'static':

            tcenter = int(inputtime)
            tcenterraw = 1
            boxtstart = inputtime
            if config['yolo_v0'] == False:
                confidence = prediction_vector[total_classes + config['c'] + b * total_coords]
            if config['yolo_v0']:
                confidence = 1

        xcentermean = xcentermean + xcenter
        ycentermean = ycentermean + ycenter
        heightmean = heightmean + height
        widthmean = widthmean + width
        confidencemean = confidencemean + confidence
        tcentermean = tcentermean + tcenter
        boxtstartmean = boxtstartmean + boxtstart
        boxtcentermean = boxtcentermean + boxtcenter
        anglemean = anglemean + angle
        xcenterrawmean = xcenterrawmean + xcenterraw
        ycenterrawmean = ycenterrawmean + ycenterraw
        tcenterrawmean = tcenterrawmean + tcenterraw
    xcentermean = xcentermean / nboxes
    ycentermean = ycentermean / nboxes
    heightmean = heightmean / nboxes
    widthmean = widthmean / nboxes
    confidencemean = confidencemean / nboxes
    tcentermean = tcentermean / nboxes
    anglemean = anglemean / nboxes
    boxtcentermean = boxtcentermean / nboxes
    xcenterrawmean = xcenterrawmean / nboxes
    ycenterrawmean = ycenterrawmean / nboxes
    tcenterrawmean = tcenterrawmean / nboxes
    boxtstartmean = boxtstartmean / nboxes

    classybox = {}
        
       
    if event_type == 'dynamic':
        if mode == 'detection':
            real_time_event = tcentermean
            box_time_event = boxtcentermean
        if mode == 'prediction':
            real_time_event = int(inputtime)
            box_time_event = int(inputtime)
        if config['yolo_v2']:
            realangle = math.pi * (anglemean + 0.5)
            rawangle = anglemean
        else:
            realangle = 2
            rawangle = 2
        # Compute the box vectors
        if marker_tree is not None:
            nearest_location = get_nearest(marker_tree, ycentermean, xcentermean, real_time_event)
            if nearest_location is not None:
               ycentermean, xcentermean = nearest_location
        box = {'xstart': xstart, 'ystart': ystart, 'tstart': boxtstartmean, 'xcenterraw': xcenterrawmean,
                'ycenterraw': ycenterrawmean, 'tcenterraw': tcenterrawmean, 'xcenter': xcentermean,
                'ycenter': ycentermean, 'real_time_event': real_time_event, 'box_time_event': box_time_event,
                'height': heightmean, 'width': widthmean, 'confidence': confidencemean, 'realangle': realangle,
                'rawangle': rawangle}
        if event_type == 'static':
            real_time_event = int(inputtime)
            box_time_event = int(inputtime)
            realangle = 0
            rawangle = 0

    

        # Make a single dict object containing the class and the box vectors return also the max prob label
        for d in [Class, box]:
            classybox.update(d)
        
        return classybox    
    


def focpredictionloop(j, k, sx, sy, nboxes, stride, time_prediction, config, key_categories, key_cord, inputz):
    total_classes = len(key_categories)
    y = (k - 1) * stride
    x = (j - 1) * stride
    prediction_vector = time_prediction[0, k - 1, j - 1, :]

    xstart = x + sx
    ystart = y + sy
    Class = {}
    # Compute the probability of each class
    for (event_name, event_label) in key_categories.items():
        Class[event_name] = prediction_vector[event_label]

    trainshapex = config['imagex']
    trainshapey = config['imagey']

    xcentermean = xstart + 0.5 * trainshapex
    ycentermean = ystart + 0.5 * trainshapey

    heightmean = trainshapey
    widthmean = trainshapex
    zcentermean = int(inputz)
    confidencemean = prediction_vector[total_classes + config['c']]

    max_prob_label = np.argmax(prediction_vector[:total_classes])

    if max_prob_label >= 0:

        real_z_event = zcentermean

        # Compute the box vectors
        box = {'xstart': xstart, 'ystart': ystart, 'xcenter': xcentermean, 'ycenter': ycentermean,
               'real_z_event': real_z_event,
               'height': heightmean, 'width': widthmean, 'confidence': confidencemean}

        # Make a single dict object containing the class and the box vectors return also the max prob label
        classybox = {}
        for d in [Class, box]:
            classybox.update(d)

        return classybox


def get_nearest(marker_tree, ycenter, xcenter, tcenter):
    location = (ycenter, xcenter)
    tree, indices = marker_tree[str(int(round(tcenter)))]
    distance, nearest_location = tree.query(location)

    if distance <= 30:
        nearest_location = int(indices[nearest_location][0]), int(indices[nearest_location][1])
        return nearest_location[0], nearest_location[1]


def draw_labelimages(image, location):
    cv2.circle(image, location, 2, (255, 0, 0), thickness=-1)

    return image


def zero_pad(image, TrainshapeX, TrainshapeY):
    time = image.shape[0]
    sizeY = image.shape[2]
    sizeX = image.shape[1]

    sizeXextend = sizeX
    sizeYextend = sizeY

    while sizeXextend % TrainshapeX != 0:
        sizeXextend = sizeXextend + 1

    while sizeYextend % TrainshapeY != 0:
        sizeYextend = sizeYextend + 1

    extendimage = np.zeros([time, sizeXextend, sizeYextend])

    extendimage[0:time, 0:sizeX, 0:sizeY] = image

    return extendimage


def twod_zero_pad(image, PadX, PadY):
    sizeY = image.shape[1]
    sizeX = image.shape[0]

    sizeXextend = sizeX
    sizeYextend = sizeY

    while sizeXextend % PadX != 0:
        sizeXextend = sizeXextend + 1

    while sizeYextend % PadY != 0:
        sizeYextend = sizeYextend + 1

    extendimage = np.zeros([sizeXextend, sizeYextend])

    extendimage[0:sizeX, 0:sizeY] = image

    return extendimage


def extra_pad(image, patchX, patchY):
    extendimage = np.zeros([image.shape[0], image.shape[1] + patchX, image.shape[2] + patchY])

    extendimage[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]] = image

    return extendimage


def save_labelimages(save_dir, image, axes, fname, Name):
    imwrite((save_dir + Name + '.tif'), image)

def Distance(centerAx, centerAy, centerBx, centerBy):
    
    distance = (centerAx - centerBx) * (centerAx - centerBx) + (centerAy - centerBy) * (centerAy - centerBy)
    
    return distance

def Genericdist(vec1, vec2):
    
    distance = 0
    for i in range(0, len(vec1)):
        distance = distance + (vec1[i] - vec2[i]) * (vec1[i] - vec2[i])
    return distance
def save_csv(save_dir, Event_Count, Name):
    Event_data = []

    Path(save_dir).mkdir(exist_ok=True)

    for line in Event_Count:
        Event_data.append(line)
    writer = csv.writer(open(save_dir + "/" + (Name) + ".csv", "w"))
    writer.writerows(Event_data)
