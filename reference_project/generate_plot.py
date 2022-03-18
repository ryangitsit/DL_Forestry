import matplotlib.pyplot as plt
import pandas as pd
import logger
from argument_parser import setup_argument_parser
import os
import seaborn as sns
from matplotlib.lines import Line2D

log = logger.setup_logger(__name__)

def main():
    config = setup_argument_parser()
    if config.crossvalidation:
        plot_crossvalidation(config)
    else:
        plot(config)
    
    

def plot_crossvalidation(config):
    fig, ax = plt.subplots()
    palette = sns.color_palette(n_colors=5)

    for kfold in range(5):
            # Plot lines for training and validation accuracies for all folds:
            fp = f"model_training_history/Foldnr{kfold+1}_{config.epochs}Epochs_{config.activation}Activation-{config.optimizer}\
Optimizer-{config.augmentation}Augmentation_lrate{config.learningrate}_mom{config.momentum}-history.csv"
            try:
                data = pd.read_csv(fp, sep = ',')
            except:
                log.error(f"kfold crossvalidation history csv files were not successfully opened (file: {fp})")
            

            accuracy = data["accuracy"]
            val_accuracy = data["val_accuracy"]
            epochs_range = range(len(data.index))
            ax.plot(epochs_range, accuracy, label="training accuracy", linestyle="dotted",color = palette[kfold])
            ax.plot(epochs_range, val_accuracy, label="validation accuracy", linestyle="dashed", color = palette[kfold])

    ax.set(
        xlabel="epoch", ylabel="Accuracy", title="Training and Validation Accuracy"
    )
    if config.optimizer == 'rms':
        optimstr = "RMSprop"
    else:
        optimstr = "SGD"
        
    caption = f"5-fold crossvalidation with settings: Activation: {config.activation},\n\
Optimizer: {optimstr}, Learning rate: {config.learningrate}, Momentum: {config.momentum}"
    fig.text(0.5, 0.02, caption, ha="center", style="italic")

    #custom legend:
    custom_lines = [Line2D([0], [0], color='grey', lw=4, linestyle = 'dotted'),
                    Line2D([0], [0], color='grey', lw=4, linestyle = 'dashed')]
    plt.legend(custom_lines, ['Training', 'Validation'], loc="upper left")
    ax.grid()
    fig.subplots_adjust(bottom=0.2)

    try:
        os.mkdir(f"./crossvalidation_results/plots")
    except:
        log.info("Folder plots already existed")
    try:
        os.mkdir(f"./crossvalidation_results/plots/{config.activation}_activation_{config.optimizer}_optimizer")
    except:
        log.info("Folder already existed")

    fig.savefig(f"./crossvalidation_results/plots/{config.activation}_activation_{config.optimizer}_optimizer/{config.epochs}Epochs_{config.activation}Activation-{config.optimizer}\
Optimizer-{config.augmentation}Augmentation_lrate{config.learningrate}_mom{config.momentum}.jpg")


def plot(config):
    # data
    fp = f"model_training_history/{config.epochs}Epochs_{config.activation}Activation-{config.optimizer}\
Optimizer-{config.augmentation}Augmentation_lrate{config.learningrate}_mom{config.momentum}-history.csv"

    try:
        data = pd.read_csv(fp, sep = ',')
    except:
        log.error(f"history csv files were not successfully opened (filepath: {fp})")

    epochs_range = range(len(data.index))
    accuracy = data["accuracy"]
    val_accuracy = data["val_accuracy"]

    # create plot
    fig, ax = plt.subplots()

    # apply datak
    accuracy = data["accuracy"]
    val_accuracy = data["val_accuracy"]
    epochs_range = range(len(data.index))
    ax.plot(epochs_range, accuracy, label="training accuracy", linestyle="dotted")
    ax.plot(epochs_range, val_accuracy, label="validation accuracy", linestyle="dashed")

    # draw boilerplate
    ax.set(
        xlabel="epoch", ylabel="Accuracy", title="Training and Validation Accuracy"
    )
    if config.optimizer == 'rms':
        optimstr = "RMSprop"
    else:
        optimstr = "SGD"

    caption = f"Accuracies with settings: Activation: {config.activation}, Optimizer: {optimstr}, \n\
Learning rate: {config.learningrate}, Momentum: {config.momentum}, Augmentation = {config.augmentation}"
    fig.text(0.5, 0.02, caption, ha="center", style="italic")

    plt.legend(loc="upper left")

    # stylize
    ax.grid()
    fig.subplots_adjust(bottom=0.2)
    # show and export
    try:
        os.mkdir("./results/")
    except:
        log.info('results folder already existed')
    fig.savefig(f"./results/{config.epochs}Epochs_{config.activation}Activation-{config.optimizer}\
    Optimizer-{config.augmentation}Augmentation_lrate{config.learningrate}_mom{config.momentum}.jpg")


if __name__ == "__main__":
    main()
