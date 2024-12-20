import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
import numpy as np
import time
from contextlib import contextmanager

# lock can be provided if there is multi processing, if not use a dummy lock
@contextmanager
def dummy_lock():
    yield  # Does nothing, placeholder for the lock


# Custom Dataset class for HDF5 (to be able to open h5py file with dataloader)
class HDF5Dataset(Dataset):
    def __init__(self, filepath, indices, lock):
        self.filepath = filepath
        self.indices = indices
        self.lock = lock
        with self.lock:
            # use a memmap for easy reading from file while training
            self.file = h5py.File(filepath, 'r')  # Open file
            self.features = self.file['X']
            self.labels = self.file['y']

    def __len__(self):
        return len(self.indices)
    #gets batch
    def __getitem__(self, idx):
        index = self.indices[idx]
        with self.lock:
            X = self.features[index]
            y = self.labels[index]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    #cleanup, close file
    def __del__(self):
        if hasattr(self, 'file') and self.file:
            self.file.close()

# Batch training function
def batch_training(model, loss_func, filepath, train_indices, lock=None, custom_print=print, batch_size=1024, epochs=5):
    # print("Max index:", max(train_indices))
    """_summary_

    Args:
        model (model): model to train (like LR XGBoost etc.)
        loss_func (func):loss function to calc new weights
        filepath (string):file with the train data
        train_indices (int): indices of samples to train o from data file
        lock (multiprocess lock): if there are multi process, use lock to read safely from files
        custom_print (func):in case of multi processing, custom print prints out the process first.
        batch_size (int, optional): batch size Defaults to 1028.
        epochs (int, optional): num epochs Defaults to 5.

    Returns:
        _type_: _description_
    """

      # Use dummy lock if no lock is provided
    if lock is None:
        lock = dummy_lock()

    

    # Create DataLoader for training
    dataset = HDF5Dataset(filepath, train_indices, lock)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    try:
        for epoch in range(epochs):
            start_time = time.time()  # Start the timer
            custom_print(f"Started epoch: {epoch + 1}")
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            batch_count = 0

            for X_batch, y_batch in dataloader:
                # Convert tensors to numpy arrays
                # custom_print(f"batch number {batch_count}")
                X_batch = X_batch.numpy()
                y_batch = y_batch.numpy()

                # Train model on current batch
                model.partial_fit(X_batch, y_batch, classes=[0,1])
                custom_print(f"finshed batch: {batch_count}")

                # Calculate loss and accuracy for this batch
                y_pred = model.predict(X_batch)
                batch_loss = loss_func(y_batch, model.predict_proba(X_batch))
                batch_accuracy = accuracy_score(y_batch, y_pred)

                # Accumulate loss and accuracy
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
                batch_count += 1

            # Average loss and accuracy for the epoch
            avg_loss = epoch_loss / batch_count
            avg_accuracy = epoch_accuracy / batch_count
            custom_print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
            end_time = time.time()  # End the timer
            epoch_duration = end_time - start_time  # Calculate duration
            custom_print(f"Epoch completed in {epoch_duration:.2f} seconds.")

    finally:
        del dataset  # Ensure resources are cleaned up

    return model
