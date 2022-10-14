# oneat

oneat = Open Network for Event as Action Topologies

[![PyPI version](https://img.shields.io/pypi/v/oneat.svg)](https://pypi.org/project/oneat)


This project provides static and action classification networks for LSTM/CNN based networks to recoganize cell events such as division, apoptosis, cell rearrangement for various imaging modalities.



## Installation & Usage

## Installation
This package can be installed by 


`pip install --user oneat`

If you are building this from the source, clone the repository and install via

```bash
git clone https://github.com/Kapoorlabs-caped/caped-ai-oneat/

cd caped-ai-oneat

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
git clone https://github.com/Kapoorlabs-caped/caped-ai-oneat/
cd caped-ai-oneat
pipenv sync

# make the current package available
pipenv run python setup.py develop

# you can run the example notebooks by starting the jupyter notebook inside the virtual env
pipenv run jupyter notebook
```

## Examples

oneat comes with different options to combine segmentation with classification or to just use classification independently of any segmentation during the model prediction step. We summarize this in the table below:

| Example Dataset   | DataSet | Trained Model | Notebook Code |
| --- |--- | --- |--- |
| <img src="https://github.com/Kapoorlabs-CAPED/CAPED-AI-oneat/blob/main/images/Xenopus_example.jpg"  title="Xenopus nuclei in 3D/4D" width="200">| [Example timelapse](https://zenodo.org/record/6484966/files/C1-for_oneat_prediction.tif)| [Oneat model]() |  [Napari notebook]()|
|   |   |  | | 
## Troubleshooting & Support

- The [image.sc forum](https://forum.image.sc/tag/oneat) is the best place to start getting help and support. Make sure to use the tag `oneat`, since we are monitoring all questions with this tag.
- If you have technical questions or found a bug, feel free to [open an issue](https://github.com/Kapoorlabs-CAPED/CAPED-AI-oneat/issues).

