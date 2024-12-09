import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
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

# Convert byte strings in 'info' to normal strings
info_str = [target.decode('utf-8') for target in info]

# Target name for testing
test_target = "GTCACCAATCCTGTCCCTAGNGG"
def run_training_LOO_with_target(test_target, num_model):
    # Print the unique values in info to debug the issue
    #print(f"Unique targets in info: {np.unique(info_str)}")

    # Check if the test_target exists in info_str
    test_indices = np.where(np.array(info_str) == test_target)[0]

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
            train_indices = np.where(np.array(info_str) != test_target)[0]
            X_train = X[train_indices]
            y_train = y[train_indices]

            # Efficient check for train/test set correctness
            if test_target in np.array(info_str)[train_indices]:
                print(f"Error: Train set contains target {test_target}, which should be left out.")
            else:
                print(f"Train set correctly does not contain the target {test_target}.")
            if not np.all(np.array(info_str)[test_indices] == test_target):
                print(f"Error: Test set contains targets other than {test_target}.")
            else:
                print(f"Test set correctly only contains the target {test_target}.")

            # Train logistic regression model
            print("Start training:")
            model = LogisticRegression(penalty='l2', C=0.5, solver='saga', max_iter=10, class_weight='balanced', verbose=1)
            model.fit(X_train, y_train)
            print("Finished training")

            # Test the model
            if X_test.shape[0] > 0:
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)

                # Calculate AUPR and AUROC
                proba_test = model.predict_proba(X_test)[:, 1]  # Probabilities for positive class
                aupr = average_precision_score(y_test, proba_test)
                auroc = roc_auc_score(y_test, proba_test)

                # Output results
                print(f"Test Guide: {test_target}")
                print(f"Number of Test Samples: {len(y_test)}")
                print(f"Accuracy: {accuracy:.2f}")
                print(f"AUPR: {aupr:.4f}")
                print(f"AUROC: {auroc:.4f}")
            else:
                print(f"Error: Test set is empty, skipping prediction step.")

            # Save the trained model to a file
            joblib.dump(model, f'models/logistic_regression_model_{num_model}.pkl')


targets1 = [
    "GGGAACCCAGCGAGTGAAGANGG",
    "GGTGAGGGAGGAGAGATGCCNGG",
    "GCGCCGAGAAGGAAGTGCTCNGG",
    "GTCCCCTCCACCCCACAGTGNGG",
    "GGGGCCACTAGGGACAGGATNGG",
    "GTCCCTAGTGGCCCCACTGTNGG",
    "GAGCCACATTAACCGGCCCTNGG",
    "GCTCGGGGACACAGGATCCCNGG",
    "GGAAGGAGGAGGCCTAAGGANGG",
    "GCAAACATGCTGTCCTGAAGNGG",
    "GGCCCCACTGTGGGGTGGAGNGG",
    "GCTGTCCTGAAGTGGACATANGG",
    "GGAATCTGCCTAACAGGAGGNGG",
    "GGCCGAGATGTCTCGCTCCGNGG",
    "GCTACTCTCTCTTTCTGGCCNGG",
    "GAGTAGCGCGAGCACAGCTANGG",
    "GGCCACGGAGCGAGACATCTNGG",
    "GAAGTTGACTTACTGAAGAANGG",
    "GCATACTCATCTTTTTCAGTNGG",
    "GGCATACTCATCTTTTTCAGNGG",
    "GGCAGAAACCCTGGTGGTCGNGG",
    "GATTTCCTCCTCGACCACCANGG",
    "GGATTTCCTCCTCGACCACCNGG",
    "GGGTATTATTGATGCTATTCNGG",
    "GATGCTATTCAGGATGCAGTNGG",
    "GGCAGCATAGTGAGCCCAGANGG",
    "GACATTAAAGATAGTCATCTNGG",
    "GGTGACAAGTGTGATCACTTNGG",
    "GCTGTGTTTGCGTCTCTCCCNGG",
    "GTATGGAAAATGAGAGCTGCNGG",
    "GTGAGTAGAGCGGAGGCAGGNGG",
    "GTAGAGCGGAGGCAGGAGGCNGG",
    "GCTGCCGCCCAGTGGGACTTNGG",
    "GCAGCATAGTGAGCCCAGAANGG",
    "GGACAGTAAGAAGGAAAAACNGG",
    "GGTACCTATCGATTGTCAGGNGG",
    "GCACGTGGCCCAGCCTGCTGNGG",
    "GAGGTTCACTTGATTTCCACNGG",
    "GGCCAGTACCACAGCAGGCTNGG",
    "GCACAAGGCTCAGCTGAACCNGG",
    "GGGACTCTACATCTGCAAGGNGG",
    "GGCCCAGCCTGCTGTGGTACNGG",
    "GTGGTACTGGCCAGCAGCCGNGG",
    "GTGTGTGAGTATGCATCTCCNGG",
    "GCTTCGGCAGGCTGACAGCCNGG"
]
targets2 = [
    "GTGCGGCAACCTACATGATGNGG",
    "GCTGGCGATGCCTCGGCTGCNGG",
    "GCAGATGGAATCATCTAGGANGG",
    "GGACTGAGGGCCATGGACACNGG",
    "GATAACTACACCGAGGAAATNGG",
    "GCCGTGGCAAACTGGTACTTNGG",
    "GTACAGGCTGCACCTGTCAGNGG",
    "GCATTTTCTTCACGGAAACANGG",
    "GAAGATGATGGAGTAGATGGNGG",
    "GGGCAATGGATTGGTCATCCNGG",
    "GAAGCATGACGGACAAGTACNGG",
    "GAAGCGTGATGACAAAGAGGNGG",
    "GAAGAAACTGAGAAGCATGANGG",
    "GTCCCCTGAGCCCATTTCCTNGG",
    "GAGGGCTCACCAGAGGTAGGNGG",
    "GCTGACCCCGCTGGGCAGGCNGG",
    "GGGGCAGCTCCGGCGCTCCTNGG",
    "GTGGAGCGCAGTGGTCTCCGNGG",
    "GCTGCCCCGCCTGCCCAGCGNGG",
    "GATGTGGGAGGCTCAGTTCCNGG",
    "GGGCTGCAGGGGAGCTGGGCNGG",
    "GCTGTTTCTGCAGCCGCTTTNGG",
    "GCTGCAGAAACAGCAAGCCCNGG",
    "GCCTCTCCAGCCAGGGGCTGNGG",
    "GGTCCCGGTGGTGTGGGCCCNGG",
    "GGTGGTGTGGGCCCAGGAGGNGG",
    "GCTGGGCAGGAGCCCCCTCCNGG",
    "GGGGGATTGTGGGGCTGCAGNGG",
    "GAAGGCTGAGATCCTGGAGGNGG",
    "GGCGCCCTGGCCAGTCGTCTNGG",
    "GGAGAAGGTGGGGGGGTTCCNGG",
    "GAAGGTGGCGTTGTCCCCTTNGG",
    "GAGAAGGTGGGGGGGTTCCANGG",
    "GCGTGACTTCCACATGAGCGNGG",
    "GGACCGCAGCCAGCCCGGCCNGG",
    "GTTGGAGAAGCTGCAGGTGANGG",
    "GCTTGTCCGTCTGGTTGCTGNGG",
    "GCCCTGGCCAGTCGTCTGGGNGG",
    "GTCTGGGCGGTGCTACAACTNGG",
    "GGGCGGTGCTACAACTGGGCNGG",
    "GCCCTGCTCGTGGTGACCGANGG",
    "GGTCACCACGAGCAGGGCTGNGG",
    "GGGGGGTTCCAGGGCCTGTCNGG",
    "GAGCAGGGCTGGGGAGAAGGNGG",
    "GGGGGTTCCAGGGCCTGTCTNGG",
    "GGAAACTTGGCCACTCTATGNGG",
    "GGCACCAACTGGATGGATCANGG",
    "GTCTCCCTGATCCATCCAGTNGG",
    "GGTGGATGATGGTGCCGTCGNGG",
    "GGGATCAGGTGACCCATATTNGG",
    "GATTTCTATGACCTGTATGGNGG",
    "GGTTTCACCGAGACCTCAGTNGG",
    "GAGACCCTGCTCAAGGGCCGNGG",
    "GATGCAGAGACCCTGCTCAANGG",
    "GGGGATTTCTATGACCTGTANGG",
    "GTTTGCGACTCTGACAGAGCNGG",
    "GTGGGATCGGAGCAGTTCAGNGG",
    "GTCAGGGTTCTGGATATCTGNGG",
    "GCTGGTACACGGCAGGGTCANGG",
    "GAGAATCAAAATCGGTGAATNGG",
    "GACACCTTCTTCCCCAGCCCNGG",
    "GTCGAGAAAAGCTTTGAAACNGG",
    "GAACAAGGTGTTCCCACCCGNGG",
    "GGTGCACAGTGGGGTCAGCANGG"
]
# remember to comment out run_training_LOO_eith_target(target, num_model) in first loop
# num_model = 2
# for target in targets1:
#     run_training_LOO_with_target(target, num_model)
#     num_model += 1
# for target in targets2:
#     run_training_LOO_with_target(target, num_model)
#     num_model += 1
run_training_LOO_with_target(test_target, 1)