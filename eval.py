import os 
import argparse 
import tensorflow as tf 

from build_dataset import load_data 


def evaluate(args):
    """
    Evaluate the model 
    """ 
    _, _, X_test, y_test = load_data(args.datapath)

    model = tf.keras.models.load_model(args.model)

    model.evaluate(X_test, y_test, verbose=1)

    test_predictions = model.predict(X_test).flatten()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        type=str,
                        default='./trained_models/model/',
                        help="Saved model location")
    parser.add_argument('--datapath',
                        type=str,
                        default='./data/',
                        help='Path to directory containing test and training data')

    
    args = parser.parse_args() 

    assert(os.path.exists(args.datapath))
    assert(os.path.exists(args.model))

    evaluate(args)