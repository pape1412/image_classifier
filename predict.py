# ++++++++++++++++++++ #
# Run image prediction #
# ++++++++++++++++++++ #

# Import libraries
import argparse

# Custom libraries
import utilities as uu
import model as mm


# Create parser for command line arguments
# Parser object and arguments
parser = argparse.ArgumentParser(description = 'PyTorch Image Classification: Class Prediction')
parser.add_argument('image_path',
                    metavar = 'PATH',
                    help = 'path to image you want to classifiy')
parser.add_argument('checkpoint_path',
                    metavar = 'PATH',
                    help = 'path to model checkpoint')
parser.add_argument('--top_k',
                    metavar = 'K',
                    default = 1,
                    help = 'number of top k predicted classes // default = 1')
parser.add_argument('--category_names',
                    metavar = 'PATH',
                    default = None,
                    help = 'path to category names in order to map predicted classes to real names // default = none')
parser.add_argument('--gpu',
                    action = 'store_true',
                    dest = 'gpu',
                    default = False,
                    help = 'enable GPU prediction // default = false')


def main():
    global args
    args = parser.parse_args()
    
    # Load model checkpoint
    model, _, _ = mm.load_checkpoint(args.checkpoint_path)
    # Pre-process input
    image = uu.process_image(args.image_path)
    # Predict image
    probs, classes = mm.predict(image,
                                model,
                                topk = args.top_k,
                                category_names = args.category_names,
                                gpu = args.gpu
                               )
    # Return results
    return probs, classes


if __name__ == '__main__':
    main()