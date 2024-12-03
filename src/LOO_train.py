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

# # Load the files
# info_df = pd.read_csv(info_file)
# vector_df = pd.read_csv(vector_file)

# Open the HDF5 file and load the embeddings
with h5py.File(hdf5_file, 'r') as f:
    # Load the dataset (X and y)
    X = f['X'][:]
    y = f['y'][:]
    info = f['info'][:]

# Target name for testing
test_target = "AAVS1_site_1"

# Filter rows for the test target
test_indices = np.where(info == test_target)[0]
X_test = X[test_indices]
y_test = y[test_indices]

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
model = LogisticRegression(max_iter=100, solver='saga', verbose=1)  # Increase iterations if needed
model.fit(X_train, y_train)
print("Finished training")

# Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Output results
print(f"Test Guide: {test_target}")
print(f"Number of Test Samples: {len(y_test)}")
print(f"Accuracy: {accuracy:.2f}")

# Save the trained model to a file
joblib.dump(model, 'logistic_regression_model_1.pkl')
