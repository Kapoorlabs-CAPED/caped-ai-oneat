import setuptools
from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()


setup(
    name="oneat",

    version='2.0.2',

    author='Varun Kapoor',
    author_email='randomaccessiblekapoor@gmail.com',
    url='https://github.com/Kapoorlabs-CAPED/CAPED-AI-oneat/',
    description='Static and Dynamic classification tool.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        
        "pandas",
        "vollseg",
        "numpy==1.20.0",
        "scipy",
        "tifffile",
        "matplotlib",
        "imagecodecs",
        "napari[all]",
        "diplib",
        "opencv-python" 
       
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
    ],
)
