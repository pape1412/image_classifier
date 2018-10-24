# ++++++++++++++++++++++++++++++++++++++ #
# Functions & classes for building model #
# ++++++++++++++++++++++++++++++++++++++ #

# Import libraries
import argparse
import numpy as np
import json
import os
import time

import torch
import torch.nn.functional as F
#from torchvision import datasets, transforms, models
from torchvision import models
#from torch.utils.data import DataLoader
from torch import nn
from torch import optim

# Define class for model classifier
class Classifier(nn.Module):
    ''' Creates deep learning model to use as classifier/fc
        in pre-trained PyTorch model
    '''
    
    # Define network architechture
    def __init__(self, num_inputs, num_outputs, num_hidden):
        super().__init__()
        
        # Hidden layers
        if num_hidden is not None:
            self.hidden_layers = nn.ModuleList([nn.Linear(num_inputs, num_hidden[0])])
            hidden_sizes = zip(num_hidden[:-1], num_hidden[1:])
            self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in hidden_sizes])
        
            # Output
            self.output = nn.Linear(num_hidden[-1], num_outputs)
        else:
            # Output
            self.output = nn.Linear(num_inputs, num_outputs)
    
    # Define forward pass
    def forward(self, x):
        try:
            for linear in self.hidden_layers:
                x = F.relu(linear(x))
            x = self.output(x)
        except AttributeError:
            x = self.output(x)
        
        return F.log_softmax(x, dim = 1)
    
# Create model from pre-trained architecture
def load_architecture(architecture, num_outputs, num_hidden = None):
    ''' Loads model architecture of pre-trained PyTorch model
        and changes the models classifier/fc object based on
        classes in given data set and hidden layers
    '''
    
    # Load model
    print('Loading architecture of pre-trained {:s} model ...'.format(architecture))
    model = models.__dict__[architecture](pretrained = True)
    
    # Freeze parameters of pretrained model
    for param in model.parameters():
        param.requires_grad = False

    # Get number of classifier input features & change classifier (start with default)
    if architecture.startswith('vgg'):
        num_inputs = model.classifier[0].in_features
        model.classifier = Classifier(num_inputs, num_outputs, num_hidden)
    elif architecture.startswith('alex'):
        num_inputs = model.classifier[1].in_features
        model.classifier = Classifier(num_inputs, num_outputs, num_hidden)
    elif architecture.startswith('dense'):
        num_inputs = model.classifier.in_features
        model.classifier = Classifier(num_inputs, num_outputs, num_hidden)
    elif architecture.startswith('incep') or architecture.startswith('res'):
        num_inputs = model.fc.in_features
        model.fc = Classifier(num_inputs, num_outputs, num_hidden)     
    
    return model


# Define optimizer
def load_optimizer(model, architecture, learning_rate):
    ''' Loads optimizer for model training based on
        architecture of pre-trained model
    '''
    
    if architecture.startswith('alex') or architecture.startswith('dense') or architecture.startswith('vgg'):
        optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    else:
        optimizer = optim.Adam(model.fc.parameters(), lr = learning_rate)
    
    return optimizer


# Define network training function
def run_training(model, train_loader, valid_loader, criterion, optimizer, num_epochs, gpu = False):
    ''' Runs deep learning model training on training data and reports
        loss and accuracy on training as well as validation data
    '''
    
    start_time = time.time()
    training_steps = 0
    #num_epochs = int(num_epochs)
    
    # Change model to CUDA (if available)
    if gpu:
        torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch_device.type == 'cpu':
            print('Warning: GPU training not available.')
    else:
        torch_device = torch.device("cpu")
    model.to(torch_device)

    # Run training
    print('Starting model training on {:s} ...'.format(torch_device.type.upper()))
    model.train()
    
    for e in range(num_epochs):
        
        running_loss = 0.0
        running_corrects = 0
        running_totals = 0
        
        for ii, (images, labels) in enumerate(train_loader):
            training_steps += 1
            
            # Change inputs/labels to CUDA (if available, see above)
            images, labels = images.to(torch_device), labels.to(torch_device)

            # Set gradients back to zero
            optimizer.zero_grad()

            # Forward and backward passes
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # Calculate training loss & accuracy on current batch
            running_loss += loss.item()
            _, preds = torch.max(output, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            running_totals += labels.size(0)

            # Report progress every 50 iterations
            if training_steps % 25 == 0:
                
                # Set model to validation mode (done in validation method) & evaluate current performance
                valid_loss, valid_accuracy = run_validation(model, valid_loader, criterion, torch_device)
                
                # Print current performance
                print("Epoch: {}/{} ...".format(
                    e + 1, num_epochs))
                print(" Train Loss: {:.3f} Train Accuracy: {:.3f}".format(
                    running_loss / 25, running_corrects / running_totals))
                print(" Valid Loss: {:.3f} Valid Accuracy: {:.3f}\n".format(
                    valid_loss, valid_accuracy))

                # Set running variables of current iteration back to 0
                running_loss = 0.0
                running_corrects = 0
                running_totals = 0

                # Set model back to training mode (just to be sure, but is also done in "run_validation" method)
                model.train()
    
    # Training duration
    train_time = time.time() - start_time
    print('Training complete.\n Total training time: {:.0f}m {:.0f}s'.format(
        train_time // 60, train_time % 60))
    
    # Return model
    return model


# Define validation function for training
def run_validation(model, valid_loader, criterion, torch_device):
    ''' Runs validation pass on full set of testing data
        and returns loss and accuracy
    '''
    
    loss = 0
    total = 0
    correct = 0
    
    # Change model to training device
    model.to(torch_device)
    
    # Set model to evaluation mode & turn off gradients for validation
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            
            # Change inputs/labels to CUDA (if available)
            images, labels = images.to(torch_device), labels.to(torch_device)
            
            # Forward pass
            output = model.forward(images)
            
            # Calculate loss on current batch
            loss += criterion(output, labels).item()
            
            # Calculate number of correctly predicted labels & batch size
            _, preds = torch.max(output.data, 1)
            #correct += (predicted == labels).sum().item()
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
    
    # Set model back to training mode 
    model.train()
    
    return (loss / len(valid_loader)), (correct / total)


# Save trained model as checkpoint
def save_checkpoint(model, optimizer, epochs, class_to_idx, save_dir):
    ''' Saves checkpoint model and parameters necessary for rebuilding
        model in order to resume training or to use for inference
    '''
    
    # Attach mapping of classes to indeces
    model.class_to_idx = class_to_idx

    # Save checkpoint
    checkpoint = {
        'model' : model,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer,
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'epochs': epochs
    }
    
    save_dir = save_dir + '/model_checkpoint.pth'
    print('Saving model checkpoint at {:s} ...'.format(save_dir))
    torch.save(checkpoint, save_dir)


# Define function to loads a checkpoint and rebuild model
def load_checkpoint(checkpoint_path):
    ''' Loads checkpoint model from file path and rebuilds model '''
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location = lambda storage, loc: storage)

    # Re-define model & freeze parameters
    model = checkpoint['model']
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load remaining information
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epochs']
    
    print('Loading model checkpoint from \'{:s}\' ...'.format(checkpoint_path))
    return model, optimizer, epochs


# Define function for predicting image class(es)
def predict(image, model, topk = 1, category_names = None, gpu = False):
    ''' Predict the class (or classes) of a pre-pocessed image
        using a trained deep learning model.
    '''
    
    # Pre-process image
    image = torch.from_numpy(image).float()
    image.unsqueeze_(0)
    
    # Change model to CUDA (if available)
    if gpu:
        torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch_device.type == 'cpu':
            print('Warning: GPU prediction not available.')
    else:
        torch_device = torch.device("cpu")
    model.to(torch_device)
    image = image.to(torch_device)

    # Run class prediction with forward pass
    print('Starting prediction on {:s} ...'.format(torch_device.type.upper()))
    # Turn off gradients for validation, saves memory and computations
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
    
    # Get probabilities from log probabilities
    probs = torch.exp(output).cpu()
    
    # Extract top K probabilities and corresponding indeces
    probs, labels = probs.topk(int(topk))
    #probs, labels = probs.numpy().tolist()[0], labels.numpy().tolist()[0]
    probs, labels = probs.numpy()[0], labels.numpy().tolist()[0]

    # Invert key value pairs from class_to_idx and save classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[key] for key in labels]
    
    # Return top K probabilites and corresponding classes
    # If mapping file is given return real names instead of classes
    if category_names is not None:
        # Get real names
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        flowers = [cat_to_name[c] for c in classes]

        # Return top K probabilites and corresponding names
        print('Probabilities:', *probs, sep = ', ')
        print('Flowers:', *flowers, sep = ', ')
        return probs, flowers
    else:
        # Return top K probabilites and corresponding classes
        print('Probabilities:', *probs, sep = ' ')
        print('Flowers:', *classes, sep = ' ')
        return probs, classes