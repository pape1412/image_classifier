# ++++++++++++++++++++++++++++++++++++++++++++ #
# Run classifier training of pre-trained model #
# ++++++++++++++++++++++++++++++++++++++++++++ #

# Import libraries
import argparse
import os
import re
from torchvision import models
from torch import nn

# Custom libraries
import utilities as uu
import model as mm


# Create parser for command line arguments
# Names of architectures to choose from (NOTE: omitted squeezenet models because unsure how to change classifier)
model_names = sorted(model for model in models.__dict__
                     if model.islower()
                     and (bool(re.search(r'\d', model)) or model == 'alexnet')
                     and model.find('squeeze') == -1)

# Parser object and arguments
parser = argparse.ArgumentParser(description = 'PyTorch Image Classification: Model Training')
parser.add_argument('data_dir',
                    metavar = 'DIR',
                    help = 'directory of training & validation data')
parser.add_argument('--save_dir',
                    metavar = 'DIR',
                    default = os.getcwd(),
                    help = 'directory of model checkpoint // default = current wd')
parser.add_argument('--arch',
                    metavar = 'ARCH',
                    default = 'vgg13',
                    choices = model_names,
                    help = 'model architectures: ' + ", ".join(model_names) + ' // default = vgg13')
parser.add_argument('--hidden_units',
                    metavar = 'H',
                    default = None,
                    type = int,
                    nargs = "+",
                    help = 'hidden units // default = None')
parser.add_argument('--learning_rate',
                    metavar = 'LR',
                    default = 0.01,
                    type = float,
                    help = 'learning rate // default = 0.01')
parser.add_argument('--epochs',
                    metavar = 'E',
                    default = 1,
                    type = int,
                    help = 'epochs // default = 1')
parser.add_argument('--gpu',
                    action = 'store_true',
                    dest = 'gpu',
                    default = False,
                    help = 'enable GPU training // default = false')

def main():
    global args
    args = parser.parse_args()
    
    # Import datasets & dataloaders
    train_data, valid_data, train_loader, valid_loader = uu.load_datasets(args.data_dir)
    # Load architechture of pre-trained model with new classifier
    num_outputs = len(train_data.classes)
    model = mm.load_architecture(args.arch,
                                 num_outputs = num_outputs,
                                 num_hidden = args.hidden_units
                                )
    # Define criterion & optimizer
    criterion = nn.NLLLoss()
    optimizer = mm.load_optimizer(model = model,
                                  architecture = args.arch,
                                  learning_rate = args.learning_rate
                                 )
    # Train model
    model_trained = mm.run_training(model = model,
                                    train_loader = train_loader,
                                    valid_loader = valid_loader,
                                    criterion = criterion,
                                    optimizer = optimizer,
                                    num_epochs = args.epochs,
                                    gpu = args.gpu
                                   )
    # Save model checkpoint
    mm.save_checkpoint(model = model_trained,
                       optimizer = optimizer,
                       epochs = args.epochs,
                       class_to_idx = train_data.class_to_idx,
                       save_dir = args.save_dir
                      )

    
if __name__ == '__main__':
    main()