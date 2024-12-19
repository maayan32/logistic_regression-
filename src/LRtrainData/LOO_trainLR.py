import h5py
import numpy as np
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import batch_train as bt
import multiprocessing


# File paths
hdf5_file = "data\data.h5"
hdf5_copy = "data\dataCopy.h5"

# Target name for testing
print("Finished extracting")
test_target = "GTCACCAATCCTGTCCCTAGNGG"

def run_training_LOO_with_target(test_target, num_model, hdf5_file,  batch_size=1028, epochs=5):
    """ 
    Args:
        test_target (string): target seq to leave out from train
        num_model (int): num model out of the 110 models in loo
        batch_size (int, optional): _description_. Defaults to 1028.
        epochs (int, optional): _description_. Defaults to 5.
    """
    # get info from file, and check wihch indices  have the target to be excluded from train
    with h5py.File(hdf5_file, 'r') as f:
        info_str = np.array(f['info']).astype(str)        
        test_indices = np.where(np.array(info_str) == test_target)[0]
        all_indices = np.arange(len(info_str))
        train_indices = all_indices[~np.isin(all_indices, test_indices)]
        del info_str

    if len(test_indices) == 0:
        print(f"Error: {test_target} not found in 'info'. Please choose a valid target.")
        return
    # Initialize the SGDClassifier and loss function:
    model = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.5, max_iter=1, warm_start=True)
    loss_func = log_loss
    print("Start training:")
    model = bt.batch_training(model, loss_func, hdf5_file, train_indices, batch_size, epochs)
    print("Finished training")
    #load the test data
    with h5py.File(hdf5_file, 'r') as f:
        features = f['X']
        labels = f['y']
        X_test = features[test_indices]
        y_test = labels[test_indices]

    if X_test.shape[0] > 0:
        #get evaluations
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        proba_test = model.predict_proba(X_test)[:, 1]  # Probabilities for positive class
        aupr = average_precision_score(y_test, proba_test)
        auroc = roc_auc_score(y_test, proba_test)

        print(f"Test Guide: {test_target}")
        print(f"Number of Test Samples: {len(y_test)}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"AUPR: {aupr:.4f}")
        print(f"AUROC: {auroc:.4f}")
    else:
        print(f"Error: Test set is empty, skipping prediction step.")
    # create obj to save info into model file
    metadata = {
        'epochs': epochs,
        'batch_size': batch_size,
        'target_name': test_target,
        'auroc': auroc,
        'aupr': aupr
    }
    #save model to file
    joblib.dump({
        'model': model,
        'metadata': metadata
    }, f'models/logistic_regression_model_{num_model}.pkl')

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
# Example usage
run_training_LOO_with_target(test_target, 1, hdf5_file)
# # Function to run the training in parallel with a limit of 2 processes at a time
# def run_parallel_training(targets, num_model, hdf5_file):
#     with multiprocessing.Pool(processes=2) as pool:  # Only 2 processes at a time
#         tasks = [(target, num_model + i, hdf5_file) for i, target in enumerate(targets)]
#         pool.starmap(run_training_LOO_with_target, tasks)  # Run the tasks in parallel

# # Example usage for running the training with two sets of targets in parallel
# if __name__ == "__main__":
#     run_parallel_training(targets1, 1, hdf5_file)
