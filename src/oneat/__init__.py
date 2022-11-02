try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
from csbdeep.utils.tf import keras_import

from oneat.NEATModels import NEATLRNet, NEATResNet, NEATTResNet, NEATVollNet
from oneat.pretrained import (
    clear_models_and_aliases,
    register_aliases,
    register_model,
)

get_file = keras_import("utils", "get_file")


clear_models_and_aliases(NEATVollNet, NEATLRNet, NEATTResNet, NEATResNet)

register_model(
    NEATLRNet,
    "Cellsplitdetectorbrightfield",
    "https://zenodo.org/record/6481021/files/Cellsplitdetectorbrightfield.h5",
    "0c6ba49c1ba0eb91819af40460fb66cb",
    "Cellsplitcordhelaflou",
    "https://zenodo.org/record/6481021/files/Cellsplitcordhelaflou.json",
    "aed21cb69d6fb8be32c47f78a39d32f5",
    "Cellsplitcategorieshelaflou",
    "https://zenodo.org/record/6481021/files/Cellsplitcategorieshelaflou.json",
    "7a67a83f08fb1add3c1b1a3e0eeec773",
    "Cellsplitdetectorbrightfield_Parameter",
    "https://zenodo.org/record/6481021/files/Cellsplitdetectorbrightfield_Parameter.json",
    "266e3e7fa7587d53d62ae0492482a1bc",
)


register_model(
    NEATVollNet,
    "Cellsplitdetectorhdpc",
    "https://zenodo.org/record/6483483/files/Cellsplitdetectorhdpc.h5",
    "701d7fc8494b66a73b024f3d3c3d3dad",
    "Cellsplitcordhelaflou",
    "https://zenodo.org/record/6481021/files/Cellsplitcordhelaflou.json",
    "aed21cb69d6fb8be32c47f78a39d32f5",
    "Cellsplitcategorieshelaflou",
    "https://zenodo.org/record/6481021/files/Cellsplitcategorieshelaflou.json",
    "7a67a83f08fb1add3c1b1a3e0eeec773",
    "Cellsplitdetectorhdpc_Parameter",
    "https://zenodo.org/record/6483483/files/Cellsplitdetectorhdpc_Parameter.json",
    "556f0ce063a4cacda6b9f27ae7aae69b",
)

register_model(
    NEATTResNet,
    "Cellsplitdetectorxenopus",
    "https://zenodo.org/record/6484966/files/Cellsplitdetectorxenopus.h5",
    "299a2edebf217da76732fd812cb5d6fe",
    "Cellsplitcordxenopus",
    "https://zenodo.org/record/6484966/files/Cellsplitcordxenopus.json",
    "aed21cb69d6fb8be32c47f78a39d32f5",
    "Cellsplitcategoriesxenopus",
    "https://zenodo.org/record/6484966/files/Cellsplitcategoriesxenopus.json",
    "7a67a83f08fb1add3c1b1a3e0eeec773",
    "Cellsplitdetectorxenopus_Parameter",
    "https://zenodo.org/record/6484966/files/Cellsplitdetectorxenopus_Parameter.json",
    "ee3dfb4ff1af80e44ceaebbdf4b1bff1",
)
register_model(
    NEATResNet,
    "Cellsplitdetectorxenopus",
    "https://zenodo.org/record/6484966/files/Cellsplitdetectorxenopus.h5",
    "299a2edebf217da76732fd812cb5d6fe",
    "Cellsplitcordxenopus",
    "https://zenodo.org/record/6484966/files/Cellsplitcordxenopus.json",
    "aed21cb69d6fb8be32c47f78a39d32f5",
    "Cellsplitcategoriesxenopus",
    "https://zenodo.org/record/6484966/files/Cellsplitcategoriesxenopus.json",
    "7a67a83f08fb1add3c1b1a3e0eeec773",
    "Cellsplitdetectorxenopus_Parameter",
    "https://zenodo.org/record/6484966/files/Cellsplitdetectorxenopus_Parameter.json",
    "ee3dfb4ff1af80e44ceaebbdf4b1bff1",
)

register_aliases(
    NEATLRNet, "Cellsplitdetectorbrightfield", "Cellsplitdetectorbrightfield"
)
register_aliases(NEATVollNet, "Cellsplitdetectorhdpc", "Cellsplitdetectorhdpc")
register_aliases(
    NEATTResNet, "Cellsplitdetectorxenopus", "Cellsplitdetectorxenopus"
)
register_aliases(
    NEATResNet, "Cellsplitdetectorxenopus", "Cellsplitdetectorxenopus"
)

del register_model, register_aliases, clear_models_and_aliases


def abspath(path):
    import os

    base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, path)


def test_image_brightfield(target):
    import os

    from tifffile import imread, imwrite

    url = "https://zenodo.org/record/6480142/files/20210904_TL2%20-%20R05-C03-F0_ch_2.tif"
    hash = "67e13fa4df301dfe2c2a57f785aedada"
    fname = abspath(get_file(fname="brightfield", origin=url, file_hash=hash))
    image = imread(fname)
    Name = os.path.basename(os.path.splitext(fname)[0])
    imwrite(target + "/" + Name + ".tif", image.astype("float32"))
    return fname


def test_image_hdpc(target):
    import os

    from tifffile import imread, imwrite

    url = "https://zenodo.org/record/6483483/files/20210904_TL2%20-%20R05-C03-F0_ch_1.tif"
    hash = "ec6fde9a27a627866e88d44a6d98f41e"
    fname = abspath(get_file(fname="hdpc", origin=url, file_hash=hash))
    image = imread(fname)
    Name = os.path.basename(os.path.splitext(fname)[0])
    imwrite(target + "/" + Name + ".tif", image.astype("float32"))
    return fname


def test_image_xenopus(target):
    import os

    from tifffile import imread, imwrite

    url = "https://zenodo.org/record/6484966/files/C1-for_oneat_prediction.tif"
    hash = "ccf1836e66c350f90a3f458890d16220"
    fname = abspath(get_file(fname="xenopus", origin=url, file_hash=hash))
    image = imread(fname)
    Name = os.path.basename(os.path.splitext(fname)[0])
    imwrite(target + "/" + Name + ".tif", image.astype("float32"))
    return fname
