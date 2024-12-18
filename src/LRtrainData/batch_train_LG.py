import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import SGDClassifier
import numpy as np
import time

# Custom Dataset class for HDF5 (to be able to open h5py file with dataloader)
class HDF5Dataset(Dataset):
    def __init__(self, filepath, indices):
        self.filepath = filepath
        self.indices = indices
        # use a memmap for easy reading from file while training
        self.file = h5py.File(filepath, 'r')  # Open file
        self.features = self.file['X']
        self.labels = self.file['y']

    def __len__(self):
        return len(self.indices)
    #gets batch
    def __getitem__(self, idx):
        index = self.indices[idx]
        X = self.features[index]
        y = self.labels[index]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    #cleanup, close file
    def __del__(self):
        if hasattr(self, 'file') and self.file:
            self.file.close()

# Batch training function
def batch_training(filepath, train_indices, batch_size=1028, epochs=5):
    # print("Max index:", max(train_indices))

    # Initialize the SGDClassifier
    model = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.5, max_iter=1, warm_start=True)

    # Create DataLoader for training
    dataset = HDF5Dataset(filepath, train_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    try:
        for epoch in range(epochs):
            start_time = time.time()  # Start the timer
            print(f"Started epoch: {epoch + 1}")
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            batch_count = 0

            for X_batch, y_batch in dataloader:
                # Convert tensors to numpy arrays
                # print(f"batch number {batch_count}")
                X_batch = X_batch.numpy()
                y_batch = y_batch.numpy()

                # Train model on current batch
                model.partial_fit(X_batch, y_batch, classes=[0,1])

                # Calculate loss and accuracy for this batch
                y_pred = model.predict(X_batch)
                batch_loss = log_loss(y_batch, model.predict_proba(X_batch))
                batch_accuracy = accuracy_score(y_batch, y_pred)

                # Accumulate loss and accuracy
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
                batch_count += 1

            # Average loss and accuracy for the epoch
            avg_loss = epoch_loss / batch_count
            avg_accuracy = epoch_accuracy / batch_count
            print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
            end_time = time.time()  # End the timer
            epoch_duration = end_time - start_time  # Calculate duration
            print(f"Epoch completed in {epoch_duration:.2f} seconds.")

    finally:
        del dataset  # Ensure resources are cleaned up

    return model
