# cyclegan

### Data downloading:

The folder BC_team_domain_experiment has been zipped and uploaded to box in case we need it for google colab. To get it, run

`!wget -O data.zip https://duke.box.com/shared/static/e25jyupdx5jlsl7d3bskshhdegpuvitu.zip ; unzip data.zip ; rm data.zip`

in one of the code blocks in the iPython notebook. The exclamation mark in the front tells the jupyter notebook that this whole thing is a linux shell command and should be interpreted as such.

### Running cyclegan non-interactively

First create a conda environment using the packages in `tf-gpu-env.yml`, and then run train.py after modifying the necessary parameters (most likely the input and output datasets and the number of training epochs).
