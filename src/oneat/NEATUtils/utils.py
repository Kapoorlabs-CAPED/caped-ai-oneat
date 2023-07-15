import collections
import csv
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import spatial
from scipy.ndimage import binary_fill_holes, find_objects
from skimage import measure, morphology
from skimage.measure import label
from skimage.morphology import binary_dilation, dilation, square
from skimage.util import invert as invertimage
from tifffile import imwrite
from tqdm import tqdm
from skimage.measure import regionprops
from tifffile import imread

def location_map(
    event_locations_dict: dict,
    seg_image: np.ndarray,
    heatmapsteps: int,
    display_3d: bool = True,
):
    cell_count = {}
    location_image = np.zeros(seg_image.shape)
    j = 0
    for i in range(seg_image.shape[0]):
        current_seg_image = seg_image[i, :]
        waterproperties = measure.regionprops(current_seg_image)
        indices = [prop.centroid for prop in waterproperties]
        cell_count[int(i)] = len(indices)

        if int(i) in event_locations_dict.keys():
            currentindices = event_locations_dict[int(i)]

            if len(indices) > 0:
                tree = spatial.cKDTree(indices)
                if len(currentindices) > 0:
                    for j in range(0, len(currentindices)):
                        index = currentindices[j]
                        closest_marker_index = tree.query(index)
                        if display_3d:
                            current_seg_label = current_seg_image[
                                int(indices[closest_marker_index[1]][0]),
                                int(indices[closest_marker_index[1]][1]),
                                int(indices[closest_marker_index[1]][2]),
                            ]
                        else:
                            current_seg_label = current_seg_image[
                                int(indices[closest_marker_index[1]][0]),
                                int(indices[closest_marker_index[1]][1]),
                            ]
                        if current_seg_label > 0:
                            all_pixels = np.where(
                                current_seg_image == current_seg_label
                            )
                            all_pixels = np.asarray(all_pixels)
                            if display_3d:
                                for k in range(all_pixels.shape[1]):
                                    location_image[
                                        i,
                                        all_pixels[0, k],
                                        all_pixels[1, k],
                                        all_pixels[2, k],
                                    ] = 1
                            else:
                                for k in range(all_pixels.shape[1]):
                                    location_image[
                                        i, all_pixels[0, k], all_pixels[1, k]
                                    ] = 1

    location_image = average_heat_map(location_image, heatmapsteps)

    return location_image, cell_count


def average_heat_map(image, sliding_window):

    j = 0
    for i in range(image.shape[0]):

        j = j + 1
        if i > 0:
            image[i, :] = np.add(image[i, :], image[i - 1, :])
        if j == sliding_window:
            image[i, :] = np.subtract(image[i, :], image[i - 1, :])
            j = 0
    return image


def remove_big_objects(ar, max_size=6400, connectivity=1, in_place=False):
    out = ar.copy()
    ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_big = component_sizes > max_size
    too_big_mask = too_big[ccs]
    out[too_big_mask] = 0

    return out


def IntergerLabelGen(BinaryImage, Name, savedir):
    InputBinaryImage = BinaryImage.astype("uint8")
    IntegerImage = np.zeros(
        [BinaryImage.shape[0], BinaryImage.shape[1], BinaryImage.shape[2]]
    )
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

    imwrite(savedir + Name + ".tif", IntegerImage.astype("uint16"))


def MarkerToCSV(MarkerImage):
    MarkerImage = MarkerImage.astype("uint16")
    MarkerList = []
    print("Obtaining co-ordinates of markers in all regions")
    for i in range(0, MarkerImage.shape[0]):
        waterproperties = measure.regionprops(MarkerImage, MarkerImage)
        indices = [prop.centroid for prop in waterproperties]
        MarkerList.append([i, indices[0], indices[1]])
    return MarkerList


def load_json(fpath):
    with open(fpath) as f:
        return json.load(f)


def save_json(data, fpath, **kwargs):
    with open(fpath, "w") as f:
        f.write(json.dumps(data, **kwargs))


def normalizeFloatZeroOne(
    x, pmin=1, pmax=99.8, axis=None, eps=1e-20, dtype=np.uint8
):
    """Percentile based Normalization

    Normalize patches of image before feeding into the network

    Parameters
    ----------
    x : np array Image patch
    pmin : minimum percentile value for normalization
    pmax : maximum percentile value for normalization
    axis : axis along which the normalization has to be carried out
    eps : avoid dividing by zero
    """
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, eps=1e-20, dtype=np.uint8):

    x = x.astype(dtype)
    mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
    ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
    eps = dtype(eps) if np.isscalar(eps) else eps.astype(dtype, copy=False)

    x = (x - mi) / (ma - mi + eps)

    return x


def generate_membrane_locations(membranesegimage : np.ndarray, csvfile: str, savefile: str):
    
    dataset = pd.read_csv(csvfile, delimiter = ',')
    
    
    writer = csv.writer(open(savefile, "a", newline=""))
    writer.writerow(
                        [
                            "T",
                            "Z",
                            "Y",
                            "X",
                            "Score",
                            "Size",
                            "Confidence",
                        ]
    )
    nrows = len(dataset.columns)
    dict_membrane = {}
    if isinstance(membranesegimage, str):
        membranesegimage = imread(membranesegimage)

    for i in tqdm(range(membranesegimage.shape[0])):
        currentimage = membranesegimage[i,:,:,:]
        properties = measure.regionprops(currentimage) 
        membrane_coordinates = [prop.centroid for prop in properties]
        dict_membrane[i] =  membrane_coordinates
        
    for index, row in dataset.iterrows():
        time = int(row[0])
        z = int(row[1])
        y = int(row[2])
        x = int(row[3])
        index = (z,y,x)
        if time < membranesegimage.shape[0]:
            if nrows > 4:
                        score = row[4]
                        size = row[5]
                        confidence = row[6]
            else:
                        score = 1.0
                        size = 10
                        confidence = 1.0
            membrane_coordinates = dict_membrane[time]
            if len(membrane_coordinates) > 0:
                tree = spatial.cKDTree(membrane_coordinates)  
                distance, nearest_location = tree.query(index)          
                            
                z = int(membrane_coordinates[nearest_location][0])         
                y = membrane_coordinates[nearest_location][1]
                x = membrane_coordinates[nearest_location][2]
                writer.writerow([time, z, y, x, score, size, confidence])


def load_training_data(directory, filename, axes=None, verbose=True):
    """Load training data in .npz format.
    The data file is expected to have the keys 'data' and 'label'
    """
    if directory is not None:
        npzdata = np.load(directory + filename)
    else:
        npzdata = np.load(filename)

    X = npzdata["data"]
    Y = npzdata["label"]
    Z = npzdata["label2"]

    if axes is None:
        axes = npzdata["axes"]
    axes = axes_check_and_normalize(axes)
    assert "C" in axes
    n_images = X.shape[0]
    assert X.shape[0] == Y.shape[0]
    assert 0 < n_images <= X.shape[0]

    X, Y = X[:n_images], Y[:n_images]
    channel = axes_dict(axes)["C"]

    X = move_channel_for_backend(X, channel=channel)

    axes = axes.replace("C", "")  # remove channel
    if backend_channels_last():
        axes = axes + "C"
    else:
        axes = axes[:1] + "C" + axes[1:]

    if verbose:
        ax = axes_dict(axes)
        n_train = len(X)
        image_size = tuple(X.shape[ax[a]] for a in "TZYX" if a in axes)
        n_dim = len(image_size)
        n_channel_in = X.shape[ax["C"]]

        print("number of  images:\t", n_train)

        print("image size (%dD):\t\t" % n_dim, image_size)
        print("axes:\t\t\t\t", axes)
        print("channels in / out:\t\t", n_channel_in)

    return (X, Y, Z), axes


def _raise(e):
    raise e


# https://docs.python.org/3/library/itertools.html#itertools-recipes
def consume(iterator):
    collections.deque(iterator, maxlen=0)


def axes_check_and_normalize(
    axes, length=None, disallowed=None, return_allowed=False
):
    """
    S(ample), T(ime), C(hannel), Z, Y, X
    """
    allowed = "STCZYX"
    axes = str(axes).upper()
    consume(
        a in allowed
        or _raise(
            ValueError(
                "invalid axis '{}', must be one of {}.".format(
                    a, list(allowed)
                )
            )
        )
        for a in axes
    )
    disallowed is None or consume(
        a not in disallowed or _raise(ValueError("disallowed axis '%s'." % a))
        for a in axes
    )
    consume(
        axes.count(a) == 1
        or _raise(ValueError("axis '%s' occurs more than once." % a))
        for a in axes
    )
    length is None or len(axes) == length or _raise(
        ValueError("axes (%s) must be of length %d." % (axes, length))
    )
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

    assert K.image_data_format() in ("channels_first", "channels_last")
    return K.image_data_format() == "channels_last"


def move_channel_for_backend(X, channel):
    if backend_channels_last():
        return np.moveaxis(X, channel, -1)
    else:
        return np.moveaxis(X, channel, 1)


def load_full_training_data(directory, filename, axes=None, verbose=True):
    """Load training data in .npz format.
    The data file is expected to have the keys 'data' and 'label'
    """

    if directory is not None:
        npzdata = np.load(directory + filename)
    else:
        npzdata = np.load(filename)

    X = npzdata["data"]
    Y = npzdata["label"]

    if axes is None:
        axes = npzdata["axes"]
    axes = axes_check_and_normalize(axes)
    assert "C" in axes
    n_images = X.shape[0]
    assert X.shape[0] == Y.shape[0]
    assert 0 < n_images <= X.shape[0]

    X, Y = X[:n_images], Y[:n_images]
    channel = axes_dict(axes)["C"]

    X = move_channel_for_backend(X, channel=channel)

    axes = axes.replace("C", "")  # remove channel
    if backend_channels_last():
        axes = axes + "C"
    else:
        axes = axes[:1] + "C" + axes[1:]

    if verbose:
        ax = axes_dict(axes)
        n_train = len(X)
        image_size = tuple(X.shape[ax[a]] for a in "TZYX" if a in axes)
        n_dim = len(image_size)
        n_channel_in = X.shape[ax["C"]]

        print("number of  images:\t", n_train)

        print("image size (%dD):\t\t" % n_dim, image_size)
        print("axes:\t\t\t\t", axes)
        print("channels in / out:\t\t", n_channel_in)

    return (X, Y), axes


def pad_timelapse(image, pad_width):

    zero_pad = np.zeros(
        [
            image.shape[0],
            image.shape[1] + pad_width[0],
            image.shape[2] + pad_width[1],
        ]
    )
    for i in range(0, image.shape[0]):

        zero_pad[i, :, :] = np.pad(
            image[i, :, :],
            (
                (pad_width[0] // 2, pad_width[0] // 2),
                (pad_width[1] // 2, pad_width[1] // 2),
            ),
            mode="edge",
        )

    return zero_pad


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

    region = (
        slice(0, image.shape[0]),
        slice(rowstart, endrow),
        slice(colstart, endcol),
    )

    # The actual pixels in that region.
    patch = image[region]

    return patch, rowstart, colstart


def DensityCounter(MarkerImage, TrainshapeX, TrainshapeY):
    AllDensity = {}

    for i in tqdm(range(0, MarkerImage.shape[0])):
        density = []
        location = []
        currentimage = MarkerImage[i, :].astype("uint16")
        waterproperties = measure.regionprops(currentimage, currentimage)
        indices = [prop.centroid for prop in waterproperties]

        for y, x in indices:

            crop_Xminus = x - int(TrainshapeX / 2)
            crop_Xplus = x + int(TrainshapeX / 2)
            crop_Yminus = y - int(TrainshapeY / 2)
            crop_Yplus = y + int(TrainshapeY / 2)

            region = (
                slice(int(crop_Yminus), int(crop_Yplus)),
                slice(int(crop_Xminus), int(crop_Xplus)),
            )
            crop_image = currentimage[region].astype("uint16")
            if (
                crop_image.shape[0] >= TrainshapeY
                and crop_image.shape[1] >= TrainshapeX
            ):
                waterproperties = measure.regionprops(crop_image, crop_image)

                labels = [prop.label for prop in waterproperties]
                labels = np.asarray(labels)
                density.append(labels.shape[0])
                location.append((int(y), int(x)))
        AllDensity[str(i)] = [density, location]

    return AllDensity


def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""

    # TODO: refactor 'fill_label_holes' and 'edt_prob' to share code
    def grow(sl, interior):
        return tuple(
            slice(s.start - int(w[0]), s.stop + int(w[1]))
            for s, w in zip(sl, interior)
        )

    def shrink(interior):
        return tuple(
            slice(int(w[0]), (-1 if w[1] else None)) for w in interior
        )

    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue
        interior = [
            (s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)
        ]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl, interior)] == i
        mask_filled = binary_fill_holes(grown_mask, **kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


def dilate_label_holes(lbl_img, iterations):
    lbl_img_filled = np.zeros_like(lbl_img)
    for lbl in range(np.min(lbl_img), np.max(lbl_img) + 1):
        mask = lbl_img == lbl
        mask_filled = binary_dilation(mask, iterations=iterations)
        lbl_img_filled[mask_filled] = lbl
    return lbl_img_filled


def MidSlices(Image, start_project_mid, end_project_mid, axis=1):

    SmallImage = Image.take(
        indices=range(
            Image.shape[axis] // 2 - start_project_mid,
            Image.shape[axis] // 2 + end_project_mid,
        ),
        axis=axis,
    )

    MaxProject = np.amax(SmallImage, axis=axis)

    return MaxProject


def MidSlicesSum(Image, start_project_mid, end_project_mid, axis=1):

    SmallImage = Image.take(
        indices=range(
            Image.shape[axis] // 2 - start_project_mid,
            Image.shape[axis] // 2 + end_project_mid,
        ),
        axis=axis,
    )

    MaxProject = np.sum(SmallImage, axis=axis)

    return MaxProject


def GenerateVolumeMarkers(segimage, pad_width):

    ndim = len(segimage.shape)
    # TZYX
    Markers = np.zeros(
        [
            segimage.shape[0],
            segimage.shape[1],
            segimage.shape[2] + pad_width[0],
            segimage.shape[3] + pad_width[1],
        ],
        dtype=np.uint16,
    )

    for i in tqdm(range(0, segimage.shape[0])):

        smallimage = segimage[i, :]
        newsmallimage = pad_timelapse(smallimage, pad_width)
        properties = measure.regionprops(newsmallimage.astype(np.uint16))

        Coordinates = [prop.centroid for prop in properties]
        if len(Coordinates) > 0:
            Coordinates = sorted(Coordinates, key=lambda k: [k[2], k[1], k[0]])
            Coordinates = np.asarray(Coordinates)

            coordinates_int = np.round(Coordinates).astype(int)
            markers_raw = np.zeros_like(newsmallimage)
            markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(
                len(Coordinates)
            )
            if ndim == 4:
                markers = morphology.dilation(
                    markers_raw.astype(np.uint16), morphology.ball(2)
                )

            Markers[i, :] = label(markers.astype(np.uint16))

    return Markers


def GenerateMarkers(
    segimage, start_project_mid=4, end_project_mid=4, pad_width=(0, 0)
):

    ndim = len(segimage.shape)
    Markers = np.zeros(
        [
            segimage.shape[0],
            segimage.shape[-2] + pad_width[0],
            segimage.shape[-1] + pad_width[1],
        ]
    )

    for i in tqdm(range(0, segimage.shape[0])):

        smallimage = segimage[i, :]
        newsmallimage = pad_timelapse(smallimage, pad_width)
        properties = measure.regionprops(newsmallimage.astype("uint16"))

        Coordinates = [prop.centroid for prop in properties]
        if len(Coordinates) > 0:
            if ndim == 3:
                Coordinates = sorted(Coordinates, key=lambda k: [k[1], k[0]])
            if ndim == 4:
                Coordinates = sorted(
                    Coordinates, key=lambda k: [k[2], k[1], k[0]]
                )
            Coordinates = np.asarray(Coordinates)

            coordinates_int = np.round(Coordinates).astype(int)
            markers_raw = np.zeros_like(newsmallimage)
            markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(
                len(Coordinates)
            )
            if ndim == 4:
                markers = morphology.dilation(
                    markers_raw.astype("uint16"), morphology.ball(2)
                )
            if ndim == 3:
                markers = morphology.dilation(
                    markers_raw.astype("uint16"), morphology.disk(2)
                )
            if ndim == 4:
                markers = MidSlices(
                    markers, start_project_mid, end_project_mid, axis=0
                )

            Markers[i, :] = label(markers.astype("uint16"))

    return Markers


def MakeTrees(segimage):
    AllTrees = {}
    print("Creating Dictionary of marker location for fast search")
    for i in tqdm(range(0, segimage.shape[0])):

        indices = []

        currentimage = segimage[i, :].astype("uint16")

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


def MakeForest(segimage):
    AllForest = {}
    # TZYX is the shape
    print("Creating Dictionary of marker location for fast search")
    for i in tqdm(range(0, segimage.shape[0])):

        indices = []

        currentimage = segimage[i, :].astype("uint16")
        currentimage = currentimage > 0
        currentimage = label(currentimage)
        props = measure.regionprops(currentimage)
        for prop in props:
            indices.append(
                (
                    int(prop.centroid[0]),
                    int(prop.centroid[1]),
                    int(prop.centroid[2]),
                )
            )

        if len(indices) > 0:
            tree = spatial.cKDTree(indices)

            AllForest[str(i)] = [tree, indices]

    return AllForest


def get_max_score_index(scores, threshold=0, top_k=0, descending=True):
    """Get the max scores with corresponding indicies
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


def distance(x, y):

    assert len(x) == len(y)
    assert isinstance(x, list) and isinstance(y, list)
    dist = 0
    for i in range(len(x)):
        dist = dist + (x[i] - y[i]) * (x[i] - y[i])
    dist = abs(dist)

    return dist


def compare_function_volume(box1, box2, imagex, imagey, imagez):
    w1, h1, d1 = box1["width"], box1["height"], box1["depth"]
    w2, h2, d2 = box2["width"], box2["height"], box2["depth"]
    x1 = box1["xstart"]
    x2 = box2["xstart"]

    y1 = box1["ystart"]
    y2 = box2["ystart"]

    z1 = box1["zstart"]
    z2 = box2["zstart"]

    xA = max(x1, x2)
    xB = min(x1 + w1, x2 + w2)
    yA = max(y1, y2)
    yB = min(y1 + h1, y2 + h2)
    zA = max(z1, z2)
    zB = min(z1 + d1, z2 + d2)

    if (
        abs(xA - xB) < imagex - 1
        or abs(yA - yB) < imagey - 1
        or abs(zA - zB) < imagez - 1
    ):
        intersect = max(0, xB - xA) * max(0, yB - yA) * max(0, zB - zA)

        area = h2 * w2 * d2 + h1 * w1 * d1 - intersect

        return float(np.true_divide(intersect, area))
    else:
        return -2


def compare_function(box1, box2, imagex, imagey):
    w1, h1 = box1["width"], box1["height"]
    w2, h2 = box2["width"], box2["height"]
    x1 = box1["xstart"]
    x2 = box2["xstart"]

    y1 = box1["ystart"]
    y2 = box2["ystart"]

    xA = max(x1, x2)
    xB = min(x1 + w1, x2 + w2)
    yA = max(y1, y2)
    yB = min(y1 + h1, y2 + h2)

    if abs(xA - xB) < imagex - 1 or abs(yA - yB) < imagey - 1:
        intersect = max(0, xB - xA) * max(0, yB - yA)

        area = h2 * w2 + h1 * w1 - intersect

        return float(np.true_divide(intersect, area))
    else:
        return -2


def compare_function_sec(box1, box2):

    x1center = box1["xcenter"]
    x2center = box2["xcenter"]

    y1center = box1["ycenter"]
    y2center = box2["ycenter"]

    return distance([x1center, y1center], [x2center, y2center])


def compare_function_sec_volume(box1, box2):

    x1center = box1["xcenter"]
    x2center = box2["xcenter"]

    y1center = box1["ycenter"]
    y2center = box2["ycenter"]

    z1center = box1["zcenter"]
    z2center = box2["zcenter"]

    return distance(
        [x1center, y1center, z1center], [x2center, y2center, z2center]
    )


def goodboxes(
    boxes,
    scores,
    nms_threshold,
    score_threshold,
    imagex,
    imagey,
    fidelity=1,
    nms_function="iou",
):

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
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            if nms_function == "dist":
                overlap = compare_function_sec(boxes[i], boxes[j])
                overlap_veto = nms_threshold * (
                    imagex * imagex + imagey * imagey
                )
                # if there is sufficient overlap, suppress the current bounding box
                if overlap <= overlap_veto:
                    count = count + 1
                    if count >= fidelity:

                        if boxes[i] not in Averageboxes:
                            Averageboxes.append(boxes[i])
                    suppress.append(pos)
            else:
                overlap = compare_function(boxes[i], boxes[j], imagex, imagey)
                overlap_veto = nms_threshold
                if overlap >= overlap_veto:
                    count = count + 1
                    if count >= fidelity:

                        if boxes[i] not in Averageboxes:
                            Averageboxes.append(boxes[i])
                    suppress.append(pos)

        # delete all indexes from the index list that are in the suppression list
        idxs = np.delete(idxs, suppress)
    # return only the indicies of the bounding boxes that were picked
    return Averageboxes


def volumeboxes(
    boxes,
    scores,
    nms_threshold,
    score_threshold,
    imagex,
    imagey,
    imagez,
    nms_function="iou",
):

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
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            if nms_function == "dist":
                overlap = compare_function_sec_volume(boxes[i], boxes[j])
                overlap_veto = nms_threshold * (
                    imagex * imagex + imagey * imagey + imagez * imagez
                )
                # if there is sufficient overlap, suppress the current bounding box
                if overlap <= overlap_veto:
                    count = count + 1
                    if boxes[i] not in Averageboxes:
                        Averageboxes.append(boxes[i])
                    suppress.append(pos)
            else:
                overlap = compare_function_volume(
                    boxes[i], boxes[j], imagex, imagey, imagez
                )
                overlap_veto = nms_threshold
                if overlap >= overlap_veto:
                    if boxes[i] not in Averageboxes:
                        Averageboxes.append(boxes[i])
                    suppress.append(pos)

        # delete all indexes from the index list that are in the suppression list
        idxs = np.delete(idxs, suppress)
    # return only the indicies of the bounding boxes that were picked
    return Averageboxes


def goldboxes(
    boxes,
    scores,
    nms_threshold,
    score_threshold,
    gridx,
    gridy,
    nms_function="iou",
):

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
        Averageboxes.append(boxes[i])
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            if nms_function == "dist":
                overlap = compare_function_sec(boxes[i], boxes[j])
                overlap_veto = nms_threshold * (gridx * gridx + gridy * gridy)
                # if there is sufficient overlap, suppress the current bounding box
                if overlap <= overlap_veto:

                    suppress.append(pos)
            else:
                overlap = compare_function(boxes[i], boxes[j], gridx, gridy)
                overlap_veto = nms_threshold
                if overlap >= overlap_veto:
                    suppress.append(pos)
        # delete all indexes from the index list that are in the suppression list
        idxs = np.delete(idxs, suppress)
    # return only the indicies of the bounding boxes that were picked
    return Averageboxes


def simpleaveragenms(
    boxes,
    scores,
    nms_threshold,
    score_threshold,
    event_name,
    gridx,
    gridy,
    nms_function="iou",
):
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
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # compute the ratio of overlap between the two boxes and the area of the second box
            if nms_function == "dist":
                overlap = compare_function_sec(boxes[i], boxes[j])
                overlap_veto = nms_threshold * (gridx * gridx + gridy * gridy)
                if overlap < overlap_veto:
                    boxAscore = boxes[i][event_name]
                    boxAXstart = boxes[i]["xstart"]
                    boxAYstart = boxes[i]["ystart"]
                    boxAXcenter = boxes[i]["xcenter"]
                    boxAYcenter = boxes[i]["ycenter"]
                    boxArealz = boxes[i]["real_z_event"]
                    boxAheight = boxes[i]["height"]
                    boxAwidth = boxes[i]["width"]
                    boxAconfidence = boxes[i]["confidence"]

                    boxBscore = boxes[j][event_name]
                    boxBXstart = boxes[j]["xstart"]
                    boxBYstart = boxes[j]["ystart"]
                    boxBXcenter = boxes[j]["xcenter"]
                    boxBYcenter = boxes[j]["ycenter"]
                    boxBrealz = boxes[j]["real_z_event"]
                    boxBheight = boxes[j]["height"]
                    boxBwidth = boxes[j]["width"]
                    boxBconfidence = boxes[j]["confidence"]

                    meanboxscore = (boxAscore + boxBscore) / 2
                    meanboxXstart = (boxAXstart + boxBXstart) / 2
                    meanboxYstart = (boxAYstart + boxBYstart) / 2
                    meanboxXcenter = (boxAXcenter + boxBXcenter) / 2
                    meanboxYcenter = (boxAYcenter + boxBYcenter) / 2
                    meanboxrealz = (boxArealz + boxBrealz) / 2
                    meanboxheight = (boxAheight + boxBheight) / 2
                    meanboxwidth = (boxAwidth + boxBwidth) / 2
                    meanboxconfidence = (boxAconfidence + boxBconfidence) / 2

                    newbox = {
                        "xstart": meanboxXstart,
                        "ystart": meanboxYstart,
                        "xcenter": meanboxXcenter,
                        "ycenter": meanboxYcenter,
                        "real_z_event": meanboxrealz,
                        "height": meanboxheight,
                        "width": meanboxwidth,
                        "confidence": meanboxconfidence,
                        event_name: meanboxscore,
                    }
                    suppress.append(pos)
            else:
                overlap = compare_function(boxes[i], boxes[j], gridx, gridy)
                overlap_veto = nms_threshold
                if overlap > overlap_veto:
                    boxAscore = boxes[i][event_name]
                    boxAXstart = boxes[i]["xstart"]
                    boxAYstart = boxes[i]["ystart"]
                    boxAXcenter = boxes[i]["xcenter"]
                    boxAYcenter = boxes[i]["ycenter"]
                    boxArealz = boxes[i]["real_z_event"]
                    boxAheight = boxes[i]["height"]
                    boxAwidth = boxes[i]["width"]
                    boxAconfidence = boxes[i]["confidence"]

                    boxBscore = boxes[j][event_name]
                    boxBXstart = boxes[j]["xstart"]
                    boxBYstart = boxes[j]["ystart"]
                    boxBXcenter = boxes[j]["xcenter"]
                    boxBYcenter = boxes[j]["ycenter"]
                    boxBrealz = boxes[j]["real_z_event"]
                    boxBheight = boxes[j]["height"]
                    boxBwidth = boxes[j]["width"]
                    boxBconfidence = boxes[j]["confidence"]

                    meanboxscore = (boxAscore + boxBscore) / 2
                    meanboxXstart = (boxAXstart + boxBXstart) / 2
                    meanboxYstart = (boxAYstart + boxBYstart) / 2
                    meanboxXcenter = (boxAXcenter + boxBXcenter) / 2
                    meanboxYcenter = (boxAYcenter + boxBYcenter) / 2
                    meanboxrealz = (boxArealz + boxBrealz) / 2
                    meanboxheight = (boxAheight + boxBheight) / 2
                    meanboxwidth = (boxAwidth + boxBwidth) / 2
                    meanboxconfidence = (boxAconfidence + boxBconfidence) / 2

                    newbox = {
                        "xstart": meanboxXstart,
                        "ystart": meanboxYstart,
                        "xcenter": meanboxXcenter,
                        "ycenter": meanboxYcenter,
                        "real_z_event": meanboxrealz,
                        "height": meanboxheight,
                        "width": meanboxwidth,
                        "confidence": meanboxconfidence,
                        event_name: meanboxscore,
                    }
                    suppress.append(pos)

            # if there is sufficient overlap, suppress the current bounding box

        if newbox is not None and newbox not in Averageboxes:
            Averageboxes.append(newbox)
            # delete all indexes from the index list that are in the suppression list
        idxs = np.delete(idxs, suppress)
    # return only the indicies of the bounding boxes that were picked

    meanscore = sum(d[event_name] for d in Averageboxes) / max(
        1, len(Averageboxes)
    )
    maxscore = max(d[event_name] for d in Averageboxes)

    box = {
        "real_z_event": meanboxrealz,
        "confidence": meanboxconfidence,
        event_name: meanscore,
        "max_score": maxscore,
    }

    return box, Averageboxes


def save_static(
    key_categories: dict, iou_classedboxes: dict, all_iou_classed_boxes: dict
):

    for (event_name, event_label) in key_categories.items():

        if event_label > 0:

            for (k, v) in iou_classedboxes.items():
                if k == event_name:
                    boxes = v[0]
                    if k in all_iou_classed_boxes.keys():
                        oldboxes = all_iou_classed_boxes[k]
                        boxes = boxes + oldboxes[0]

            all_iou_classed_boxes[event_name] = [boxes]
    return all_iou_classed_boxes


def save_static_csv(
    key_categories: dict, iou_classedboxes: dict, savedir: str
):

    for (event_name, event_label) in key_categories.items():

        if event_label > 0:
            xlocations = []
            ylocations = []
            scores = []
            confidences = []
            tlocations = []
            radiuses = []

            iou_current_event_boxes = iou_classedboxes[event_name][0]
            iou_current_event_boxes = sorted(
                iou_current_event_boxes,
                key=lambda x: x[event_name],
                reverse=True,
            )

            for iou_current_event_box in iou_current_event_boxes:
                xcenter = iou_current_event_box["xcenter"]
                ycenter = iou_current_event_box["ycenter"]
                tcenter = iou_current_event_box["real_time_event"]
                confidence = iou_current_event_box["confidence"]
                score = iou_current_event_box[event_name]
                radius = (
                    np.sqrt(
                        iou_current_event_box["height"]
                        * iou_current_event_box["height"]
                        + iou_current_event_box["width"]
                        * iou_current_event_box["width"]
                    )
                    // 2
                )
                radius = radius
                # Replace the detection with the nearest marker location
                xlocations.append(xcenter)
                ylocations.append(ycenter)
                scores.append(score)
                confidences.append(confidence)
                tlocations.append(tcenter)
                radiuses.append(radius)

            event_count = np.column_stack(
                [
                    tlocations,
                    ylocations,
                    xlocations,
                    scores,
                    radiuses,
                    confidences,
                ]
            )
            event_count = sorted(
                event_count, key=lambda x: x[0], reverse=False
            )
            event_data = []
            csvname = savedir + "/" + event_name + "Location"
            writer = csv.writer(open(csvname + ".csv", "a", newline=""))
            filesize = os.stat(csvname + ".csv").st_size
            if filesize < 1:
                writer.writerow(["T", "Y", "X", "Score", "Size", "Confidence"])
            for line in event_count:
                if line not in event_data:
                    event_data.append(line)
                writer.writerows(event_data)
                event_data = []


def gold_nms(
    heatmap,
    classedboxes,
    event_name,
    iou_threshold,
    event_threshold,
    gridx,
    gridy,
    generate_map,
    nms_function,
):

    sorted_event_box = classedboxes[event_name][0]
    scores = [
        sorted_event_box[i][event_name] for i in range(len(sorted_event_box))
    ]

    good_sorted_event_box = goldboxes(
        sorted_event_box,
        scores,
        iou_threshold,
        event_threshold,
        gridx,
        gridy,
        nms_function=nms_function,
    )
    for iou_current_event_box in sorted_event_box:
        xcenter = iou_current_event_box["xcenter"]
        ycenter = iou_current_event_box["ycenter"]
        tcenter = iou_current_event_box["real_time_event"]
        score = iou_current_event_box[event_name]

        if generate_map:
            for x in range(int(xcenter - 8), int(xcenter + 8)):
                for y in range(int(ycenter - 8), int(ycenter + 8)):
                    if y < heatmap.shape[1] and x < heatmap.shape[2]:
                        heatmap[int(tcenter), int(y), int(x)] = (
                            heatmap[int(tcenter), int(y), int(x)] + score
                        )

    return good_sorted_event_box


def volume_dynamic_nms(
    classedboxes,
    event_name,
    iou_threshold,
    event_threshold,
    gridx,
    gridy,
    gridz,
    nms_function="iou",
):

    sorted_event_box = classedboxes[event_name][0]
    scores = [
        sorted_event_box[i][event_name] for i in range(len(sorted_event_box))
    ]
    best_sorted_event_box = volumeboxes(
        sorted_event_box,
        scores,
        iou_threshold,
        event_threshold,
        gridx,
        gridy,
        gridz,
        nms_function=nms_function,
    )

    return best_sorted_event_box


def dynamic_nms(
    heatmap,
    classedboxes,
    event_name,
    iou_threshold,
    event_threshold,
    gridx,
    gridy,
    fidelity,
    generate_map=True,
    nms_function="iou",
):

    sorted_event_box = classedboxes[event_name][0]
    scores = [
        sorted_event_box[i][event_name] for i in range(len(sorted_event_box))
    ]
    best_sorted_event_box = goodboxes(
        sorted_event_box,
        scores,
        iou_threshold,
        event_threshold,
        gridx,
        gridy,
        fidelity=fidelity,
        nms_function=nms_function,
    )

    if generate_map:
        for iou_current_event_box in sorted_event_box:
            xcenter = iou_current_event_box["xcenter"]
            ycenter = iou_current_event_box["ycenter"]
            tcenter = iou_current_event_box["real_time_event"]
            score = iou_current_event_box[event_name]

            for x in range(int(xcenter - 8), int(xcenter + 8)):
                for y in range(int(ycenter - 8), int(ycenter + 8)):
                    if y < heatmap.shape[1] and x < heatmap.shape[2]:
                        heatmap[int(tcenter), int(y), int(x)] = (
                            heatmap[int(tcenter), int(y), int(x)] + score
                        )

    return best_sorted_event_box


def microscope_dynamic_nms(
    classedboxes,
    event_name,
    iou_threshold,
    event_threshold,
    gridx,
    gridy,
    fidelity,
    nms_function,
):

    sorted_event_box = classedboxes[event_name][0]
    scores = [
        sorted_event_box[i][event_name] for i in range(len(sorted_event_box))
    ]
    best_sorted_event_box = goodboxes(
        sorted_event_box,
        scores,
        iou_threshold,
        event_threshold,
        gridx,
        gridy,
        fidelity,
        nms_function,
    )

    return best_sorted_event_box


def save_dynamic_csv(key_categories, iou_classedboxes, savedir, z=0):

    for (event_name, event_label) in key_categories.items():

        if event_label > 0:
            xlocations = []
            ylocations = []
            zlocations = []
            scores = []
            confidences = []
            tlocations = []
            radiuses = []

            iou_current_event_boxes = iou_classedboxes[event_name][0]
            iou_current_event_boxes = sorted(
                iou_current_event_boxes,
                key=lambda x: x[event_name],
                reverse=True,
            )
            for iou_current_event_box in iou_current_event_boxes:
                xcenter = iou_current_event_box["xcenter"]
                ycenter = iou_current_event_box["ycenter"]
                tcenter = iou_current_event_box["real_time_event"]
                confidence = iou_current_event_box["confidence"]
                score = iou_current_event_box[event_name]
                radius = (
                    np.sqrt(
                        iou_current_event_box["height"]
                        * iou_current_event_box["height"]
                        + iou_current_event_box["width"]
                        * iou_current_event_box["width"]
                    )
                    // 2
                )
                radius = radius

                xlocations.append(xcenter)
                ylocations.append(ycenter)
                scores.append(score)
                confidences.append(confidence)
                tlocations.append(tcenter)
                radiuses.append(radius)
                zlocations.append(z)
            event_count = np.column_stack(
                [
                    tlocations,
                    zlocations,
                    ylocations,
                    xlocations,
                    scores,
                    radiuses,
                    confidences,
                ]
            )
            event_count = sorted(
                event_count, key=lambda x: x[0], reverse=False
            )
            event_data = []
            csvname = savedir + "/" + event_name + "Location"

            writer = csv.writer(open(csvname + ".csv", "a", newline=""))
            filesize = os.stat(csvname + ".csv").st_size

            if filesize < 1:
                writer.writerow(
                    ["T", "Z", "Y", "X", "Score", "Size", "Confidence"]
                )
            for line in event_count:
                if line not in event_data:
                    event_data.append(line)
                writer.writerows(event_data)
                event_data = []


def save_volume(
    key_categories: dict, iou_classedboxes: dict, all_iou_classed_boxes: dict
):

    for (event_name, event_label) in key_categories.items():

        if event_label > 0:

            for (k, v) in iou_classedboxes.items():
                if k == event_name:
                    boxes = v[0]
                    if k in all_iou_classed_boxes.keys():
                        oldboxes = all_iou_classed_boxes[k]
                        boxes = boxes + oldboxes[0]

            all_iou_classed_boxes[event_name] = [boxes]
    return all_iou_classed_boxes


def save_volume_csv(
    key_categories: dict, iou_classedboxes: dict, savedir: str
):

    for (event_name, event_label) in key_categories.items():

        if event_label > 0:
            xlocations = []
            ylocations = []
            zlocations = []
            scores = []
            confidences = []
            tlocations = []
            radiuses = []

            iou_current_event_boxes = iou_classedboxes[event_name][0]
            iou_current_event_boxes = sorted(
                iou_current_event_boxes,
                key=lambda x: x[event_name],
                reverse=True,
            )
            for iou_current_event_box in iou_current_event_boxes:
                xcenter = iou_current_event_box["xcenter"]
                ycenter = iou_current_event_box["ycenter"]
                zcenter = iou_current_event_box["zcenter"]
                tcenter = iou_current_event_box["real_time_event"]
                confidence = iou_current_event_box["confidence"]
                score = iou_current_event_box[event_name]
                radius = (
                    np.sqrt(
                        iou_current_event_box["height"]
                        * iou_current_event_box["height"]
                        + iou_current_event_box["width"]
                        * iou_current_event_box["width"]
                        + iou_current_event_box["depth"]
                        * iou_current_event_box["depth"]
                    )
                    // 3
                )

                xlocations.append(xcenter)
                ylocations.append(ycenter)
                zlocations.append(zcenter)
                scores.append(score)
                confidences.append(confidence)
                tlocations.append(tcenter)
                radiuses.append(radius)

            event_count = np.column_stack(
                [
                    tlocations,
                    zlocations,
                    ylocations,
                    xlocations,
                    scores,
                    radiuses,
                    confidences,
                ]
            )
            event_count = sorted(
                event_count, key=lambda x: x[0], reverse=False
            )
            event_data = []
            csvname = savedir + "/" + f"pred_{event_name}_locations"

            writer = csv.writer(open(csvname + ".csv", "a", newline=""))
            filesize = os.stat(csvname + ".csv").st_size

            if filesize < 1:
                writer.writerow(
                    ["T", "Z", "Y", "X", "Score", "Size", "Confidence"]
                )
            for line in event_count:
                if line not in event_data:
                    event_data.append(line)
                writer.writerows(event_data)
                event_data = []


def yoloprediction(
    sy,
    sx,
    time_prediction,
    stride,
    inputtime,
    config,
    key_categories,
    key_cord,
    nboxes,
    mode,
    event_type,
    marker_tree=None,
):
    LocationBoxes = []
    j = 0
    k = 1
    while True:
        j = j + 1
        if j > time_prediction.shape[1]:
            j = 1
            k = k + 1

        if k > time_prediction.shape[0]:
            break
        Classybox = predictionloop(
            j,
            k,
            sx,
            sy,
            nboxes,
            stride,
            time_prediction,
            config,
            key_categories,
            key_cord,
            inputtime,
            mode,
            event_type,
            marker_tree=marker_tree,
        )
        # Append the box and the maximum likelehood detected class
        if Classybox is not None:
            LocationBoxes.append(Classybox)

    return LocationBoxes


def volumeyoloprediction(
    sz,
    sy,
    sx,
    time_prediction,
    stride,
    inputtime,
    config,
    key_categories,
    key_cord,
    nboxes,
    mode,
    event_type,
    marker_tree=None,
):
    LocationBoxes = []
    j = 0

    k = 1
    for i in range(time_prediction.shape[0]):
        while True:
            j = j + 1

            if j > time_prediction.shape[1]:
                j = 1

                k = k + 1

            if k > time_prediction.shape[2]:
                break
            Classybox = volumepredictionloop(
                i,
                j,
                k,
                sz,
                sy,
                sx,
                nboxes,
                stride,
                time_prediction,
                config,
                key_categories,
                key_cord,
                inputtime,
                mode,
                event_type,
                marker_tree=marker_tree,
            )
            # Append the box and the maximum likelehood detected class
            if Classybox is not None:
                LocationBoxes.append(Classybox)

    return LocationBoxes


def volumepredictionloop(
    i,
    j,
    k,
    sz,
    sy,
    sx,
    nboxes,
    stride,
    time_prediction,
    config,
    key_categories,
    key_cord,
    inputtime,
    mode,
    event_type,
    marker_tree=None,
):
    total_classes = len(key_categories)
    total_coords = len(key_cord)
    y = (k - 1) * stride
    x = (j - 1) * stride
    z = (i - 1) * stride
    prediction_vector = time_prediction[i - 1, k - 1, j - 1, :]

    xstart = x + sx
    ystart = y + sy
    zstart = z + sz
    Class = {}
    # Compute the probability of each class
    for (event_name, event_label) in key_categories.items():
        Class[event_name] = prediction_vector[event_label]

    xcentermean = 0
    ycentermean = 0

    zcentermean = 0
    xcenterrawmean = 0
    ycenterrawmean = 0
    zcenterrawmean = 0

    widthmean = 0
    heightmean = 0
    depthmean = 0

    zcenter = 0
    confidencemean = 0
    trainshapex = config["imagex"]
    trainshapey = config["imagey"]
    trainshapez = config["imagez"]
    zcenterraw = 0
    for b in range(0, nboxes):
        xcenter = (
            xstart
            + prediction_vector[total_classes + config["x"] + b * total_coords]
            * trainshapex
        )
        ycenter = (
            ystart
            + prediction_vector[total_classes + config["y"] + b * total_coords]
            * trainshapey
        )
        zcenter = (
            zstart
            + prediction_vector[total_classes + config["z"] + b * total_coords]
            * trainshapez
        )
        xcenterraw = prediction_vector[
            total_classes + config["x"] + b * total_coords
        ]
        ycenterraw = prediction_vector[
            total_classes + config["y"] + b * total_coords
        ]
        zcenterraw = prediction_vector[
            total_classes + config["z"] + b * total_coords
        ]
        try:
            height = (
                prediction_vector[
                    total_classes + config["h"] + b * total_coords
                ]
                * trainshapex
            )
            width = (
                prediction_vector[
                    total_classes + config["w"] + b * total_coords
                ]
                * trainshapey
            )
            depth = (
                prediction_vector[
                    total_classes + config["d"] + b * total_coords
                ]
                * trainshapez
            )
        except ValueError:
            height = trainshapey
            width = trainshapex
            depth = trainshapez
        if event_type == "dynamic" and mode == "detection":
            confidence = prediction_vector[
                total_classes + config["c"] + b * total_coords
            ]

        if mode == "prediction":
            confidence = 1

        if event_type == "static":

            confidence = prediction_vector[
                total_classes + config["c"] + b * total_coords
            ]

        xcentermean = xcentermean + xcenter
        ycentermean = ycentermean + ycenter
        zcentermean = zcentermean + zcenter

        heightmean = heightmean + height
        widthmean = widthmean + width
        depthmean = depthmean + depth

        confidencemean = confidencemean + confidence

        xcenterrawmean = xcenterrawmean + xcenterraw
        ycenterrawmean = ycenterrawmean + ycenterraw
        zcenterrawmean = zcenterrawmean + zcenterraw

    xcentermean = xcentermean / nboxes
    ycentermean = ycentermean / nboxes
    zcentermean = zcentermean / nboxes

    heightmean = heightmean / nboxes
    widthmean = widthmean / nboxes
    depthmean = depthmean / nboxes

    confidencemean = confidencemean / nboxes
    xcenterrawmean = xcenterrawmean / nboxes
    ycenterrawmean = ycenterrawmean / nboxes
    zcenterrawmean = zcenterrawmean / nboxes

    classybox = {}

    box = None
    if event_type == "dynamic":
        if mode == "detection":
            real_time_event = inputtime
        if mode == "prediction":
            real_time_event = int(inputtime)

        # Compute the box vectors
        if marker_tree is not None:
            nearest_location = get_nearest_volume(
                marker_tree,
                zcentermean,
                ycentermean,
                xcentermean,
                real_time_event,
            )
            if nearest_location is not None:
                zcentermean, ycentermean, xcentermean = nearest_location
        # Correct for zero padding

        box = {
            "xstart": xstart,
            "ystart": ystart,
            "zstart": zstart,
            "xcenterraw": xcenterrawmean,
            "ycenterraw": ycenterrawmean,
            "zcenterraw": zcenterrawmean,
            "xcenter": xcentermean,
            "ycenter": ycentermean,
            "zcenter": zcentermean,
            "real_time_event": real_time_event,
            "height": heightmean,
            "width": widthmean,
            "depth": depthmean,
            "confidence": confidencemean,
        }

    if event_type == "static":
        real_time_event = int(inputtime)
        if marker_tree is not None:
            nearest_location = get_nearest_volume(
                marker_tree,
                zcentermean,
                ycentermean,
                xcentermean,
                real_time_event,
            )
            if nearest_location is not None:
                zcentermean, ycentermean, xcentermean = nearest_location

        box = {
            "xstart": xstart,
            "ystart": ystart,
            "zstart": zstart,
            "xcenterraw": xcenterrawmean,
            "ycenterraw": ycenterrawmean,
            "zcenterraw": zcenterrawmean,
            "xcenter": xcentermean,
            "ycenter": ycentermean,
            "zcenter": zcentermean,
            "real_time_event": real_time_event,
            "height": heightmean,
            "width": widthmean,
            "depth": depthmean,
            "confidence": confidencemean,
        }

    if box is not None:
        # Make a single dict object containing the class and the box vectors return also the max prob label
        for d in [Class, box]:
            classybox.update(d)

        return classybox


def focyoloprediction(
    sy, sx, z_prediction, stride, inputz, config, key_categories
):
    LocationBoxes = []
    j = 0
    k = 1
    while True:
        j = j + 1
        if j > z_prediction.shape[2]:
            j = 1
            k = k + 1

        if k > z_prediction.shape[1]:
            break
        Classybox = focpredictionloop(
            j, k, sx, sy, stride, z_prediction, config, key_categories, inputz
        )
        # Append the box and the maximum likelehood detected class
        if Classybox is not None:
            LocationBoxes.append(Classybox)
    return LocationBoxes


def predictionloop(
    j,
    k,
    sx,
    sy,
    nboxes,
    stride,
    time_prediction,
    config,
    key_categories,
    key_cord,
    inputtime,
    mode,
    event_type,
    marker_tree=None,
):
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
    boxtstartmean = 0
    boxtstart = inputtime
    tcenter = 0
    boxtcenter = 0
    confidencemean = 0
    trainshapex = config["imagex"]
    trainshapey = config["imagey"]
    tcenterraw = 0
    for b in range(0, nboxes):
        xcenter = (
            xstart
            + prediction_vector[total_classes + config["x"] + b * total_coords]
            * trainshapex
        )
        ycenter = (
            ystart
            + prediction_vector[total_classes + config["y"] + b * total_coords]
            * trainshapey
        )
        xcenterraw = prediction_vector[
            total_classes + config["x"] + b * total_coords
        ]
        ycenterraw = prediction_vector[
            total_classes + config["y"] + b * total_coords
        ]

        try:
            height = (
                prediction_vector[
                    total_classes + config["h"] + b * total_coords
                ]
                * trainshapex
            )
            width = (
                prediction_vector[
                    total_classes + config["w"] + b * total_coords
                ]
                * trainshapey
            )
        except ValueError:
            height = trainshapey
            width = trainshapex
        if event_type == "dynamic" and mode == "detection":
            time_frames = config["size_tminus"] + config["size_tplus"] + 1
            tcenter = int(
                inputtime
                + (
                    prediction_vector[
                        total_classes + config["t"] + b * total_coords
                    ]
                    * time_frames
                )
            )
            tcenterraw = prediction_vector[
                total_classes + config["t"] + b * total_coords
            ]
            boxtcenter = int(
                prediction_vector[
                    total_classes + config["t"] + b * total_coords
                ]
            )
            boxtstart = inputtime
            confidence = prediction_vector[
                total_classes + config["c"] + b * total_coords
            ]

        if mode == "prediction":
            confidence = 1

        if event_type == "static":

            tcenter = int(inputtime)
            tcenterraw = 1
            boxtstart = inputtime
            confidence = prediction_vector[
                total_classes + config["c"] + b * total_coords
            ]

        xcentermean = xcentermean + xcenter
        ycentermean = ycentermean + ycenter
        heightmean = heightmean + height
        widthmean = widthmean + width
        confidencemean = confidencemean + confidence
        tcentermean = tcentermean + tcenter
        boxtstartmean = boxtstartmean + boxtstart
        boxtcentermean = boxtcentermean + boxtcenter
        xcenterrawmean = xcenterrawmean + xcenterraw
        ycenterrawmean = ycenterrawmean + ycenterraw
        tcenterrawmean = tcenterrawmean + tcenterraw
    xcentermean = xcentermean / nboxes
    ycentermean = ycentermean / nboxes
    heightmean = heightmean / nboxes
    widthmean = widthmean / nboxes
    confidencemean = confidencemean / nboxes
    tcentermean = tcentermean / nboxes
    boxtcentermean = boxtcentermean / nboxes
    xcenterrawmean = xcenterrawmean / nboxes
    ycenterrawmean = ycenterrawmean / nboxes
    tcenterrawmean = tcenterrawmean / nboxes
    boxtstartmean = boxtstartmean / nboxes

    classybox = {}

    box = None
    if event_type == "dynamic":
        if mode == "detection":
            real_time_event = tcentermean
            box_time_event = boxtcentermean
        if mode == "prediction":
            real_time_event = int(inputtime)
            box_time_event = int(inputtime)

        # Compute the box vectors
        if marker_tree is not None:
            nearest_location = get_nearest(
                marker_tree, ycentermean, xcentermean, real_time_event
            )
            if nearest_location is not None:
                ycentermean, xcentermean = nearest_location
        # Correct for zero padding

        box = {
            "xstart": xstart,
            "ystart": ystart,
            "tstart": boxtstartmean,
            "xcenterraw": xcenterrawmean,
            "ycenterraw": ycenterrawmean,
            "tcenterraw": tcenterrawmean,
            "xcenter": xcentermean,
            "ycenter": ycentermean,
            "real_time_event": real_time_event,
            "box_time_event": box_time_event,
            "height": heightmean,
            "width": widthmean,
            "confidence": confidencemean,
        }

    if event_type == "static":
        real_time_event = int(inputtime)
        box_time_event = int(inputtime)
        if marker_tree is not None:
            nearest_location = get_nearest(
                marker_tree, ycentermean, xcentermean, real_time_event
            )
            if nearest_location is not None:
                ycentermean, xcentermean = nearest_location

        box = {
            "xstart": xstart,
            "ystart": ystart,
            "tstart": boxtstartmean,
            "xcenterraw": xcenterrawmean,
            "ycenterraw": ycenterrawmean,
            "tcenterraw": tcenterrawmean,
            "xcenter": xcentermean,
            "ycenter": ycentermean,
            "real_time_event": real_time_event,
            "box_time_event": box_time_event,
            "height": heightmean,
            "width": widthmean,
            "confidence": confidencemean,
        }

    if box is not None:
        # Make a single dict object containing the class and the box vectors return also the max prob label
        for d in [Class, box]:
            classybox.update(d)

        return classybox


def focpredictionloop(
    j, k, sx, sy, stride, time_prediction, config, key_categories, inputz
):
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

    trainshapex = config["imagex"]
    trainshapey = config["imagey"]

    xcentermean = xstart + 0.5 * trainshapex
    ycentermean = ystart + 0.5 * trainshapey

    heightmean = trainshapey
    widthmean = trainshapex
    zcentermean = int(inputz)
    confidencemean = prediction_vector[total_classes + config["c"]]

    max_prob_label = np.argmax(prediction_vector[:total_classes])

    if max_prob_label >= 0:

        real_z_event = zcentermean

        # Compute the box vectors
        box = {
            "xstart": xstart,
            "ystart": ystart,
            "xcenter": xcentermean,
            "ycenter": ycentermean,
            "real_z_event": real_z_event,
            "height": heightmean,
            "width": widthmean,
            "confidence": confidencemean,
        }

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
        nearest_location = int(indices[nearest_location][0]), int(
            indices[nearest_location][1]
        )
        return nearest_location[0], nearest_location[1]
    else:
        return None


def get_nearest_volume(marker_tree, zcenter, ycenter, xcenter, tcenter):
    location = (zcenter, ycenter, xcenter)
    tree, indices = marker_tree[str(int(round(tcenter)))]
    distance, nearest_location = tree.query(location)

    if distance <= 30:
        nearest_location = (
            int(indices[nearest_location][0]),
            int(indices[nearest_location][1]),
            int(indices[nearest_location][2]),
        )
        return nearest_location[0], nearest_location[1], nearest_location[2]
    else:
        return None


def save_labelimages(save_dir, image, axes, fname, Name):
    imwrite((save_dir + Name + ".tif"), image)


def Distance(centerAx, centerAy, centerBx, centerBy):

    distance = (centerAx - centerBx) * (centerAx - centerBx) + (
        centerAy - centerBy
    ) * (centerAy - centerBy)

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
    writer = csv.writer(
        open(save_dir + "/" + (Name) + ".csv", "w", newline="")
    )
    writer.writerows(Event_data)
