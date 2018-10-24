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

Also, please note that GPU training is only available if it's supported by your system (you can follow this [link](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu) for more information). 

## Files
The ```train.py``` file allows you to train a new neural network on your data set. When finished, it'll save the trained model as a checkpoint. The ```predict.py``` file predicts the class of an input image while using a pre-trained model (e.g. the checkpoint you saved after training). It should output the predicted class as well as the class probability.

Both the ```model.py & utilities.py``` files are used to organize code in the ```.py``` files for training and prediction only.

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
By design you will use transfer learning when training your model with this tool. As default it uses the pre-trained __VGG13__ architecture from PyTorch. If you're not familiar with transfer learning you can use this [link](http://cs231n.github.io/transfer-learning/) as a starting point.

The basic usage for model training is:
```$ python train.py data_dir```

While the training of the model runs you should see a command line ouput showing training and validation loss as well as validation accuracy. Despite the basic execution you can tweak your model training by using the following options:
- Set checkpoint directory: ```$ python train.py data_dir --save_dir my_dir```
- Set architechture: ```$ python train.py data_dir --arch "vgg13"```
- Set learning rate: ```$ python train.py data_dir --learning_rate 0.01```
- Set single layer hidden units: ```$ python train.py data_dir --hidden_units 512```
- Set multi layer hidden units: ```$ python train.py data_dir --hidden_units [512,512]```
- Set epochs: ```$ python train.py data_dir --epochs 20```
- Use GPU training: ```$ python train.py data_dir --gpu```
For more information on possible options please use: ```python train.py -h```

### Prediction
Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu

## Acknowledgements
