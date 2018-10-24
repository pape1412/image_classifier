# +++++++++++++++++++++++++++++++++++++ #
# Utility functions for image classifier
# +++++++++++++++++++++++++++++++++++++ #

# Import libraries
import numpy as np
import os

#import torch
#import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#from torch import nn
#from torch import optim
from PIL import Image

# Load data
def load_datasets(data_dir):
    ''' Loads training and validation data from directory, pre-processes it
        and returns data sets and corresponding dataloaders
    '''

    # File paths
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # Training set transforms
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomRotation(90),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # Validation set transforms
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    print('Loading training & validation data from \'{:s}\' ...'.format(data_dir))
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)

    # Define the dataloaders
    print('Defining dataloaders ...')
    train_loader = DataLoader(train_data, batch_size = 32, shuffle = True)
    valid_loader = DataLoader(train_data, batch_size = 32)
    
    return train_data, valid_data, train_loader, valid_loader

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Load image from file path
    image = Image.open(image_path)
    
    # Re-size image while keeping aspect ratio
    width, height = image.size
    max_pixels = max(width, height)
    if width > height:
        image.thumbnail([max_pixels, 256], Image.ANTIALIAS)
    else:
        image.thumbnail([256, max_pixels], Image.ANTIALIAS)

    # Crop out center of image
    # Python Imaging Library uses a coordinate system with (0,0) in the upper left corner
    size = 224
    width, height = image.size
    left = (width - size) / 2
    upper = (height - size) / 2
    right = (left + size)
    lower = (upper + size)
    image = image.crop((left, upper, right, lower))

    # Convert color channel to floats between 0-1
    image = np.array(image) / 255

    # Normalize image
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image = (image - means) / stds

    # Re-order dimension
    image = image.transpose(2, 0, 1)
    
    # Return image
    print('Processing image from \'{:s}\' ...'.format(image_path))          
    return image
