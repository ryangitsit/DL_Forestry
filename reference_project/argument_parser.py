import argparse

def setup_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", help = "Number of epochs to train for", type = int, default = 10)
    parser.add_argument("--activation", help = "Hidden layer activation function", choices = ['relu', 'elu'], type = str, default = 'relu')
    parser.add_argument("--optimizer", help = "Specified optimizer algorithm", choices = ['rms', 'sgdm'], type=str, default = 'rms')
    parser.add_argument("--augmentation", help = "Boolean: whether augmentation should be applied to training data", type = bool, default = False)
    parser.add_argument("--crossvalidation", help = "Apply crossvalidation and output final accuracies, rather than entire history", type = bool, default = False)
    parser.add_argument("--learningrate", help = "The learning rate specified in the respective optimizer", type = float, default = 0.045)
    parser.add_argument("--momentum", help = "Momentum specified in the respective optimizer", type = float, default = 0.0)

    return parser.parse_args()
    