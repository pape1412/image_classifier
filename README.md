# Image Classifier
Developing a command line tool for image classification.

## Intro
This tool allows you to easily build your own image classifier from the command line. By design it will use transfer learning when training your model. The default architecture is a pre-trained __VGG13__ model from PyTorch. If you're not familiar with transfer learning you can use this [link](http://cs231n.github.io/transfer-learning/) as a starting point.

## Installation
### Libraries
Despite standard libraries coming with the Miniconda distribution of Python you'll need to have ```torch & torchvision``` as well as ```PIL``` installed in order to get the command line tool up and running.

### Other Requirements
Please be aware that model training can take a very long time. In the end, this really depends on multiple factors such as:
- size of your training data
- model architecture
- hardware requirements
- ...

Also, please note that GPU training is only available if it's supported by your system (you can follow this [link](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu) for more information). 

## Files
The ```train.py``` file allows you to train a new neural network on your data set. When finished, it'll save the trained model as a checkpoint. The ```predict.py``` file predicts the class of an input image while using a pre-trained model (e.g. the checkpoint you saved after training). Once done it will output the predicted class as well as the class probability.

Both the ```model.py & utilities.py``` files are used to organize code in the ```.py``` files for training and prediction only.

## Usage
### Data
Before you can start to build you're own image classifier, you need to make sure that the __data you're feeding to the model training comes in the correct format__. You'll need to store two separate folders within one single directory - one for training, one for testing data. Inside each of these folders you need to have one additional folder with images for each class:
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
The __basic usage__ for model training is:
```$ python train.py data_dir```

While the training of the model runs you should see a command line ouput showing training and validation loss as well as validation accuracy. Despite the basic execution you can __tweak your model training__ by using the following options:
- Set __checkpoint directory__:
  
  ```$ python train.py data_dir --save_dir my_dir```
- Set __architechture__:
  
  ```$ python train.py data_dir --arch "vgg13"```
- Set __learning rate__:
  
  ```$ python train.py data_dir --learning_rate 0.01```
- Set __single layer hidden units__:
  
  ```$ python train.py data_dir --hidden_units 512```
- Set __multi layer hidden units__:
  
  ```$ python train.py data_dir --hidden_units [512,512]```
- Set __epochs__:
  
  ```$ python train.py data_dir --epochs 20```
- Use __GPU training__:
  
  ```$ python train.py data_dir --gpu```

For more information on possible options and defaults please use: ```python train.py -h```

### Prediction
The __basic usage__ for image class prediction is:
```$ python predict.py /path/to/image model_checkpoint```
As soon as the prediction is done you will see an output of the predicted class of the image as well as the class probability. Similar to model training, the prediction itself come with a couple of __options__ for you to use:
- Set __number of top classes__ to return:
  
  ```$ python predict.py /path/to/image model_checkpoint --tok_k 5```
- Set __mapping of classes to real names__:
  
  ```$ python predict.py /path/to/image model_checkpoint --category_names cat_to_name.json```
- Use __GPU for prediction__:
  
  ```$ python predict.py /path/to/image model_checkpoint --gpu```

Should you want to use a __mapping of classes__ to real names during prediction, this mapping should be provided within a ```.json``` file, e.g.:
```
{'1': 'name 1',
 '2': 'name 2',
 ...
 '9': 'name 3'}
 ```
Again, for additional information on possible options and defaults use:  ```python predict.py -h```

## Acknowledgements
This mini project was one of the projects I've worked on during my Data Scientist Nanodegree at [Udacity](https://eu.udacity.com). Also, I'd like to mention that this project would not have been possible without the extensive resources provided by the [PyTorch tutorials](https://pytorch.org/tutorials/).
