import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import zipfile
import pandas as pd

# Unzip the data
with zipfile.ZipFile("output_data_h5py/data.zip", "r") as zip_ref:
    zip_ref.extractall("data")  # Extract to 'data' directory

# File paths
hdf5_file = "data/data.h5"
info_file = "data/OTS_T_info.csv"
vector_file = "data/OTS_T_samples.csv"

# Open the HDF5 file and load the embeddings
with h5py.File(hdf5_file, 'r') as f:
    # Load the dataset (X and y)
    X = f['X'][:]
    y = f['y'][:]
    info = f['info'][:]

# Target name for testing
test_target = "GTCACCAATCCTGTCCCTAGNGG"

# Print the unique values in info to debug the issue
print(f"Unique targets in info: {np.unique(info)}")

# Check if the test_target exists in info
test_indices = np.where(info == test_target)[0]

if len(test_indices) == 0:
    print(f"Error: {test_target} not found in 'info'. Please choose a valid target.")
else:
    # Filter rows for the test target
    X_test = X[test_indices]
    y_test = y[test_indices]

    # Debugging: Print the shape of the test set
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    # Check if X_test is empty
    if X_test.shape[0] == 0:
        print(f"Error: No samples found for target {test_target} in the test set.")
    else:
        # Remaining rows for training
        train_indices = np.where(info != test_target)[0]
        X_train = X[train_indices]
        y_train = y[train_indices]

        # Efficient check for train/test set correctness
        if test_target in info[train_indices]:
            print(f"Error: Train set contains target {test_target}, which should be left out.")
        else:
            print(f"Train set correctly does not contain the target {test_target}.")
        if not np.all(info[test_indices] == test_target):
            print(f"Error: Test set contains targets other than {test_target}.")
        else:
            print(f"Test set correctly only contains the target {test_target}.")

        # Train logistic regression model
        print("Start training:")
        model = LogisticRegression(max_iter=10, solver='saga', verbose=1)  # Increase iterations if needed
        model.fit(X_train, y_train)
        print("Finished training")

        # Test the model
        if X_test.shape[0] > 0:
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            # Output results
            print(f"Test Guide: {test_target}")
            print(f"Number of Test Samples: {len(y_test)}")
            print(f"Accuracy: {accuracy:.2f}")
        else:
            print(f"Error: Test set is empty, skipping prediction step.")

        # Save the trained model to a file
        joblib.dump(model, 'logistic_regression_model_1.pkl')
