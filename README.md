# Image Classifier
Developing a command line tool for image classification.

## Installation
### Libraries
Despite standard libraries coming with the Miniconda distribution of Python you'll need to have ```torch & torchvision``` as well as ```PIL``` installed in order to get the command line tool up and running.

### Other Requirements
Please be aware that model training can take a very long time. In the end, this really depends on multiple factors such as:
- size of your training data
- model architecture
- hardware requirements
- ...

Also, please note that GPU training is only available if it's supported by your system. For refer to this [link](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu) for more information. 

## Files
```
- model.py
- predict.py
- train.py
- utilities.py
```
## Usage
### Data
This command line tools allows you to easily build your own image classifier. However, before doing so you need to make sure the data you're feeding the model training comes in the correct format. You'll need to store two separate folders within one single directory - one for training, one for testing data. Inside each of these folders you need to have one additional folder with images for each class:
```
data_dir\
  train_dir\
    0\
      image_1.jpeg
      image_2.jpeg
      ...
    1\
      ...
    ...
      ...
    9\
      ...
```
### Training

### Prediction

## Acknowledgements
