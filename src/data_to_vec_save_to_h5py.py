import pandas as pd
import numpy as np
import h5py
import os
import zipfile
import gc
import turn_sample_to_vector as ts
import orginize_df

# Process data
final_data_df, only_target_offtarget = orginize_df.proccess_data()

# Chunk size for memory management
chunk_size = 100
output_dir = os.path.join(os.path.dirname(__file__), 'output_data_h5py')
os.makedirs(output_dir, exist_ok=True)

# Define output paths for CSV files
vectors_and_labels_csv = os.path.join(output_dir, 'OTS_T_samples.csv')
final_data_csv = os.path.join(output_dir, 'OTS_T_info.csv')

# Initialize HDF5 file for embeddings
hdf5_file = os.path.join(output_dir, 'data.h5')
with h5py.File(hdf5_file, 'w') as f:
    f.create_dataset('X', shape=(0, 368), maxshape=(None, 368), dtype=np.float32)
    f.create_dataset('y', shape=(0,), maxshape=(None,), dtype=np.int32)
    f.create_dataset('info', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())

# Process data in chunks and save to HDF5
for start in range(0, len(only_target_offtarget), chunk_size):
    # Explicitly create a copy of the chunk
    chunk = only_target_offtarget.iloc[start:start + chunk_size].copy()

    # Apply function to create the embeddings for the chunk
    chunk['Embedding'] = chunk.apply(
        lambda row: ts.create_full_feature_vector(row['target'], row['offtarget_sequence']), axis=1
    )

    # Get the embeddings, labels, and info
    embeddings = np.array(chunk['Embedding'].tolist())  # Convert list of NumPy arrays to a single NumPy array
    labels = chunk['label'].values
    info = chunk['target'].values.astype(str)

    # Open HDF5 file and append the new data
    with h5py.File(hdf5_file, 'a') as f:
        f['X'].resize(f['X'].shape[0] + embeddings.shape[0], axis=0)
        f['X'][-embeddings.shape[0]:] = embeddings
        f['y'].resize(f['y'].shape[0] + labels.shape[0], axis=0)
        f['y'][-labels.shape[0]:] = labels
        f['info'].resize(f['info'].shape[0] + info.shape[0], axis=0)
        f['info'][-info.shape[0]:] = info

    # Free memory after processing chunk
    del chunk
    gc.collect()

# Save final_data_df (information about targets) to CSV
final_data_df.to_csv(final_data_csv, index=False)

# Save embeddings and labels to CSV
embeddings_df = pd.DataFrame({
    'Embedding': [embedding.tolist() for embedding in embeddings],
    'label': labels
})
embeddings_df.to_csv(vectors_and_labels_csv, index=False)

# Zip the CSV and HDF5 files
zip_path = os.path.join(output_dir, 'dataOTSforLR.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(final_data_csv, os.path.basename(final_data_csv))
    zipf.write(vectors_and_labels_csv, os.path.basename(vectors_and_labels_csv))
    zipf.write(hdf5_file, os.path.basename(hdf5_file))

# Optionally remove the CSV and HDF5 files after zipping
os.remove(final_data_csv)
os.remove(vectors_and_labels_csv)
os.remove(hdf5_file)

print(f"Zipped files created at {zip_path}")