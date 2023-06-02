import os
import pandas as pd

def split_features_labels(data, label="distance"):
    """
    Separate the label from the features 
    """

    features = data.copy() 
    labels = features.pop(label)

    return features, labels

def load_data(data_dir):
    """
    Load gaze data from specified directory
    """
    
    # load training data 
    train_path = os.path.join(data_dir, 'train.csv')
    train_data = pd.read_csv(train_path)

    # load testing data 
    test_path = os.path.join(data_dir, 'test.csv')
    test_data = pd.read_csv(test_path) 

    # split labels and features 
    train_features, train_labels = split_features_labels(train_data)
    test_features, test_labels = split_features_labels(test_data)

    return train_features, train_labels, test_features, test_labels


if __name__ == "__main__":

    X_train, y_train, X_test, y_test = load_data('data/') 

    assert(not X_train.empty)
    assert(not X_test.empty)

    assert(X_train.shape[1] == 13)
    assert(X_test.shape[1] == 13)
