import logger
from argument_parser import setup_argument_parser
from dataset import create_dataset
import inception_model as inception
import crossval as cv


log = logger.setup_logger(__name__)

def main():
    config = setup_argument_parser()
    log.info("Starting...")
    log.info("Model will train with following parameters:")
    log.info(config)

    #Create the dataset:
    if not config.crossvalidation:
        train_ds, val_ds = create_dataset(config)
        model = inception.create_model(config, train_ds, val_ds)
    else:
        # Crossvalidation comes down to using a different dataset generation system
        cv.evaluate(config)


if __name__ == "__main__":
    main()
