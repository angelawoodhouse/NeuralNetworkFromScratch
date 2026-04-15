# Load and process MNIST data
# Import packages
import numpy as np
import pandas as pd
import os

def load_and_split_data(csv_path=None, normalise=True, split_ratio=0.9):
    if csv_path is None:
        # Get directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, '..', 'data', 'train.csv')

    # Read in data & inspect head
    data = pd.read_csv(csv_path)

    # Convert to numpy array for manipulation
    data = np.array(data)

    # Get dimensions of data
    m, n = data.shape

    # Shuffle data - ensure subsets include mixes of all cases in event that original data is ordered
    np.random.shuffle(data)

    # Calculate split index
    split_idx = int(m * split_ratio)

    # Split into dev and train
    data_train = data[:split_idx]
    data_dev = data[split_idx:]

    # Process dev set
    data_dev = data_dev.T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    if normalise:
        X_dev = X_dev / 255

    # Process train set
    data_train = data_train.T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    if normalise:
        X_train = X_train / 255

    # Get training set size
    _, m_train = X_train.shape

    return X_train, Y_train, X_dev, Y_dev





