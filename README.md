# Oneat

[![PyPI version](https://img.shields.io/pypi/v/oneat.svg)](https://pypi.org/project/oneat)


This project provides static and action classification networks for LSTM based networks to recoganize cell events such as division, apoptosis, cell rearrangement for various imaging modalities.



## Installation & Usage

## Installation
This package can be installed by 


`pip install --user oneat`

additionally ensure that your installed tensorflow version is not over 2.3.4

If you are building this from the source, clone the repository and install via

```bash
git clone https://github.com/Kapoorlabs-CAPED/CAPED-AI-oneat/

cd CAPED-AI-oneat

pip install --user -e .

# or, to install in editable mode AND grab all of the developer tools
# (this is required if you want to contribute code back to NapaTrackMater)
pip install --user -r requirements.txt
```


### Pipenv install

Pipenv allows you to install dependencies in a virtual environment.

```bash
# install pipenv if you don't already have it installed
pip install --user pipenv

# clone the repository and sync the dependencies
git clone https://github.com/Kapoorlabs-CAPED/CAPED-AI-oneat/
cd CAPED-AI-oneat
pipenv sync

# make the current package available
pipenv run python setup.py develop

# you can run the example notebooks by starting the jupyter notebook inside the virtual env
pipenv run jupyter notebook
```

## Examples

oneat comes with different options to combine segmentation with classification or to just use classification independently of any segmentation during the model prediction step. We summarize this in the table below:

| Example Dataset | Description  | Trained Model | Notebook Code | Heat Map  | Csv output  | Visualization Notebook |
| --- | --- |--- | --- |--- | --- |--- |
| <img src="https://github.com/kapoorlab/vollseg-napari/blob/main/vollseg_napari/images/Ascadian_raw.png"  title="Raw Ascadian Embryo" width="200">| Light sheet fused from four angles 3D single channel| [Training Data ~320 GB](https://figshare.com/articles/dataset/Astec-half-Pm1_Cut_at_2-cell_stage_half_Phallusia_mammillata_embryo_live_SPIM_imaging_stages_6-16_/11309570?backTo=/s/765d4361d1b073beedd5)| [UNET model](https://zenodo.org/record/6337699) |<img src="https://github.com/kapoorlab/vollseg-napari/blob/main/vollseg_napari/images/Ascadian_GT.png" title="GT Ascadian Embryo" width="200"> | UNET model, slice_merge = False | [Colab Notebook](https://github.com/kapoorlab/VollSeg/blob/main/examples/Predict/Colab_VollSeg_Ascadian_Embryo.ipynb) |<img src="https://github.com/kapoorlab/vollseg-napari/blob/main/vollseg_napari/images/Ascadian_pred.png" title="Prediction Ascadian Embryo" width="200" > | <img src="https://github.com/kapoorlab/vollseg-napari/blob/main/vollseg_napari/images/Metrics_Ascadian.png" title="Metrics Ascadian Embryo" width="200" >  |

## Troubleshooting & Support

- The [image.sc forum](https://forum.image.sc/tag/oneat) is the best place to start getting help and support. Make sure to use the tag `oneat`, since we are monitoring all questions with this tag.
- If you have technical questions or found a bug, feel free to [open an issue](https://github.com/Kapoorlabs-CAPED/CAPED-AI-oneat/issues).

