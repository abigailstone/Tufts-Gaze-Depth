import os 
import argparse 

from build_dataset import load_data
from model.mlp import GazeNet 



def train_model(model, train_features, train_labels):
    """ 
    Compile and train a GazeNet model with the given training data
    """

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss
    ) 

    history = model.fit(
        train_features,
        train_labels,
        epochs=model.epochs,
        validation_split=0.2,
        verbose=1
    )

    return history


def main(args):
    """ 
    Load data, build the model, and train.
    """

    X_train, y_train, X_test, y_test = load_data(args.datapath)

    model = GazeNet() 

    history = train_model(model, X_train, y_train) 

    model.save(os.path.join(args.savepath, 'model')) 


if __name__ == "__main__":

    parser = argparse.ArgumentParser() 

    parser.add_argument('--datapath', 
                        type=str, 
                        default='./data/', 
                        help='Path to directory containing test and train data') 
    parser.add_argument('--epochs',
                        type=int, 
                        default=25,
                        help='Number of training epochs') 
    parser.add_argument('--savepath',
                        type=str,
                        default='trained_models/',
                        help="Path for saving trained model")


    args = parser.parse_args() 

    assert(os.path.exists(args.datapath))

    main(args)
