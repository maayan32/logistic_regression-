import h5py
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import batch_train as bt
import multiprocessing
from targets import targets1, targets2, target_sample_map



# File paths
hdf5_file = "data/data.h5"
# Lock for file access (for the use of multi-process)
file_lock = multiprocessing.Lock()

# Target name for testing (target number one- model number 1)
test_target = "GTCACCAATCCTGTCCCTAGNGG"

def run_training_LOO_with_target(test_target, num_model, hdf5_file,  batch_size=1024, epochs=5):

    def custom_print(*args, **kwargs):
        print(f"[Process {num_model}] ", *args, **kwargs)
    """ 
    Args:
        test_target (string): target seq to leave out from train
        num_model (int): num model out of the 110 models in loo
        batch_size (int, optional): _description_. Defaults to 1024.
        epochs (int, optional): _description_. Defaults to 5.
    """
    with file_lock:
        # get info from file, and check which indices  have the target to be excluded from train
        with h5py.File(hdf5_file, 'r') as f:
            info_str = np.array(f['info']).astype(str)        
            test_indices = np.where(np.array(info_str) == test_target)[0]
            all_indices = np.arange(len(info_str))
            train_indices = all_indices[~np.isin(all_indices, test_indices)]
            del info_str

    if len(test_indices) == 0:
        custom_print(f"Error: {test_target} not found in 'info'. Please choose a valid target.")
        return
    # Initialize the SGDClassifier and loss function:
    model = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.5, max_iter=1, warm_start=True)
    loss_func = log_loss
    custom_print("Start training:")
    model = bt.batch_training(model, loss_func, hdf5_file, train_indices, file_lock, custom_print, batch_size, epochs)
    custom_print("Finished training")
    with file_lock:
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

        custom_print(f"Test Guide: {test_target}")
        custom_print(f"Number of Test Samples: {len(y_test)}")
        custom_print(f"Accuracy: {accuracy:.2f}")
        custom_print(f"AUPR: {aupr:.4f}")
        custom_print(f"AUROC: {auroc:.4f}")
    else:
        custom_print(f"Error: Test set is empty, skipping prediction step.")
    # create obj to save info into model file
    metadata = {
        'epochs': epochs,
        'batch_size': batch_size,
        'target': test_target,
        'target_name': target_sample_map[test_target],
        'auroc': auroc,
        'aupr': aupr
    }
    with file_lock:

        #save model to file
        joblib.dump({
            'model': model,
            'metadata': metadata
        }, f'LooModels/LR_model_LOO{num_model}.pkl')

# remember to comment out run_training_LOO_eith_target(target, num_model) in first loop
# num_model = 2
# for target in targets1:
#     run_training_LOO_with_target(target, num_model)
#     num_model += 1
# for target in targets2:
#     run_training_LOO_with_target(target, num_model)
#     num_model += 1
# Example usage
# run_training_LOO_with_target(test_target, 1, hdf5_file)
# Function to run the training in parallel with alternating files
def run_parallel_training(targets, num_model, file, num_process):
    with multiprocessing.Pool(processes=num_process) as pool:  # Only num_process processes at a time
        # Alternate files in the task list
        tasks = [
            (target, num_model + i, file) 
            for i, target in enumerate(targets)
        ]
        pool.starmap(run_training_LOO_with_target, tasks)  # Run the tasks in parallel

#  Example usage for running the training with two sets of targets in parallel
if __name__ == "__main__":
    #  try_targets = [ "GGGAACCCAGCGAGTGAAGANGG",
    # "GGTGAGGGAGGAGAGATGCCNGG", "GCGCCGAGAAGGAAGTGCTCNGG","GTCCCCTCCACCCCACAGTGNGG"]

     run_parallel_training(targets1, 1, hdf5_file, 4)
