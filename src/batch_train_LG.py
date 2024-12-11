from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
import numpy as np
from sklearn.metrics import log_loss  # Import log_loss to compute the loss

def batch_training(X, y, train_indices, batch_size=512, epochs=5, validation_data=None):
    model = SGDClassifier(
        loss='log_loss', penalty='l2', alpha=0.5, max_iter=1, warm_start=True
    )
    
    n_samples = len(train_indices)  # Train only on the subset defined by train_indices

    if validation_data:
        X_val, y_val = validation_data
    else:
        X_val, y_val = None, None

    for epoch in range(epochs):
        # Shuffle the train indices for each epoch
        train_indices = shuffle(train_indices)
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = train_indices[start:end]  # Dynamically slice train_indices
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            model.partial_fit(X_batch, y_batch, classes=np.unique(y))

            # Calculate batch loss and accuracy
            #Calculate batch loss using log_loss from sklearn.metrics
            y_pred_proba = model.predict_proba(X_batch)  # Get predicted probabilities
            batch_loss = log_loss(y_batch, y_pred_proba)  # Compute log loss for the batch
            epoch_loss += batch_loss

            batch_accuracy = model.score(X_batch, y_batch)
            epoch_accuracy += batch_accuracy

        # Epoch summary
        print(f"Epoch {epoch + 1}/{epochs}: "
              f"Loss={epoch_loss / (n_samples // batch_size):.4f}, "
              f"Accuracy={epoch_accuracy / (n_samples // batch_size):.4f}")

        # Optional validation evaluation
        if X_val is not None and y_val is not None:
            val_accuracy = model.score(X_val, y_val)
            print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    return model