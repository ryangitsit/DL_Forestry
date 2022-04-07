from make_parser import setup_argument_parser
from make_data import *
import make_model as make_model
#import crossval as cv



def main():
    print("Start")
    config = setup_argument_parser()
    print("###########", config, "###########")
    # log.info("Starting...")
    # log.info("Model will train with following parameters:")
    # log.info(config)

    #Create the dataset:
    # if not config.crossvalidation:
    train_ds, val_ds = create_data()
    model = make_model.model_stuff(train_ds, val_ds, config)
    # else:
    #     # Crossvalidation comes down to using a different dataset generation system
    #     cv.evaluate(config)


if __name__ == "__main__":
    main()