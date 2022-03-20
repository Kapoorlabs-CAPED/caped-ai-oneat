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

| Example Dataset   | DataSet | Trained Model | Notebook Code | Heat Map  | Csv output  | Visualization Notebook |
| --- |--- | --- |--- | --- |--- |--- |
| <img src="https://github.com/Kapoorlabs-CAPED/CAPED-AI-oneat/blob/main/images/ch_0_crop.png"  title="Low Contrast DPC (Digital Phase Contrast)" width="200">| [Example timelapse](https://zenodo.org/record/6371249/files/20210904_TL2%20-%20R05-C03-F0_ch_0.tif)| [Oneat model]() | [Colab Notebook]() |[Heat Map]() |[Csv File]()  | [Napari notebook] ()|
| <img src="https://github.com/Kapoorlabs-CAPED/CAPED-AI-oneat/blob/main/images/ch_1_crop.png"  title="High Contrast DPC (Digital Phase Contrast)" width="200">| [Example timelapse](https://zenodo.org/record/6371249/files/20210904_TL2%20-%20R05-C03-F0_ch_1.tif)| [Oneat model]() | [Colab Notebook]() |[Heat Map]() |[Csv File]()  | [Napari notebook] ()|
| <img src="https://github.com/Kapoorlabs-CAPED/CAPED-AI-oneat/blob/main/images/ch_3_crop.png"  title="EGFP-α-tubulin" width="200">| [Example timelapse](https://zenodo.org/record/6371249/files/20210904_TL2%20-%20R05-C03-F0_ch_3.tif)| [Oneat model]() | [Colab Notebook]() |[Heat Map]() |[Csv File]()  | [Napari notebook] ()|
| <img src="https://github.com/Kapoorlabs-CAPED/CAPED-AI-oneat/blob/main/images/ch_4_crop.png"  title="mCherry-H2B" width="200">| [Example timelapse](https://zenodo.org/record/6371249/files/20210904_TL2%20-%20R05-C03-F0_ch_4.tif)| [Oneat model]() | [Colab Notebook]() |[Heat Map]() |[Csv File]()  | [Napari notebook] ()|
| <img src="https://github.com/Kapoorlabs-CAPED/CAPED-AI-oneat/blob/main/images/ch_5_crop.png"  title="Flou" width="200">| [Example timelapse](https://zenodo.org/record/6371249/files/20210904_TL2%20-%20R05-C03-F0_ch_5.tif)| [Oneat model]() | [Colab Notebook]() |[Heat Map]() |[Csv File]()  | [Napari notebook] ()|
| <img src="https://github.com/Kapoorlabs-CAPED/CAPED-AI-oneat/blob/main/images/ch_2_crop.png"  title="Brightfield" width="200">| [Example timelapse](https://zenodo.org/record/6371249/files/20210904_TL2%20-%20R05-C03-F0_ch_2.tif)| [Oneat model]() | [Colab Notebook]() |[Heat Map]() |[Csv File]()  | [Napari notebook] ()|
## Troubleshooting & Support

- The [image.sc forum](https://forum.image.sc/tag/oneat) is the best place to start getting help and support. Make sure to use the tag `oneat`, since we are monitoring all questions with this tag.
- If you have technical questions or found a bug, feel free to [open an issue](https://github.com/Kapoorlabs-CAPED/CAPED-AI-oneat/issues).

