# cyclegan

### Preface

This repository contains the code to running cyclegan on the 2021 Bass Connections team's data. It is mostly adapted from [https://www.tensorflow.org/tutorials/generative/cyclegan].

### Data downloading:

The folder `colab-cyclegan-data`, containing training images, has been zipped and uploaded to box in case we need it for google colab. To get it, run

`!wget -O data.zip https://duke.box.com/shared/static/5yfb0hgw6dphe3p9oexml8vfqncvjo2j -O data.zip ; unzip data.zip ; rm data.zip`

in one of the code blocks in the iPython notebook. The exclamation mark in the front tells the Jupyter notebook that this whole thing is a Linux shell command and should be interpreted as such. This should extract and make a folder called `colab-cyclegan-data`

<!-- The `.ipynb` notebook was used for debugging. The Python scripts adapted from the notebooks were what were run ultimately. -->
**IMPORTANT:** the predict script is not working. We used the `ipynb` file manually for predicting.

### Running cyclegan non-interactively

(Download `colab-cyclegan-data` through the method above if you haven't already)

`conda env create -f cyclegan.yml`

`conda activate cyclegan`

`python train_wrapper.py`

~~`python predict_wrapper.py`~~ `cyclegan.ipynb`
