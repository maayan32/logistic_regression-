import h5py
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
import numpy as np
from sklearn.metrics import log_loss  # Import log_loss to compute the loss

def batch_training(filepath, train_indices, batch_size=512, epochs=5):
    model = SGDClassifier(
        loss='log_loss', penalty='l2', alpha=0.5, max_iter=1, warm_start=True
    )
    n_samples = len(train_indices)  # Train only on the subset defined by train_indices

    for epoch in range(epochs):
        print(f"started epoch: {epoch}")
        # Shuffle the train indices for each epoch
        train_indices = shuffle(train_indices)
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        batch_count = 0

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = train_indices[start:end]  # Dynamically slice train_indices
            batch_indices = np.sort(batch_indices)  # Ensure indices are sorted
            with h5py.File(filepath, 'r') as f:
                features = f['X']
                labels = f['y']
                X_batch = features[batch_indices]
                y_batch = labels[batch_indices]
            model.partial_fit(X_batch, y_batch, classes=[0,1])
          # Calculate loss and accuracy for this batch
            y_pred = model.predict(X_batch)
            batch_loss = log_loss(y_batch, model.predict_proba(X_batch))
            batch_accuracy = accuracy_score(y_batch, y_pred)

            # Accumulate batch loss and accuracy
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
            batch_count += 1

        # Calculate average loss and accuracy for the entire epoch
        avg_epoch_loss = epoch_loss / batch_count
        avg_epoch_accuracy = epoch_accuracy / batch_count

        # Log average loss and accuracy for the epoch
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_epoch_loss:.4f}, Accuracy = {avg_epoch_accuracy:.4f}")
    return model