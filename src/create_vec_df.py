import orginize_df
import turn_sample_to_vector as ts
import pandas as pd
import zipfile
import os
import gc  # Garbage collection

# Process data
final_data_df, only_target_offtarget = orginize_df.proccess_data()

# Chunk size for memory management
chunk_size = 100
output_dir = os.path.join(os.path.dirname(__file__), 'output_data')
os.makedirs(output_dir, exist_ok=True)

# Define CSV output paths
vectors_and_labels_csv = os.path.join(output_dir, 'OTS_T_samples.csv')
final_data_csv = os.path.join(output_dir, 'OTS_T_info.csv')

# Create CSV file for appending chunks
with open(vectors_and_labels_csv, 'w') as f:
    pd.DataFrame(columns=['Embedding', 'label']).to_csv(f, index=False)

# Process data in chunks
for start in range(0, len(only_target_offtarget), chunk_size):
    # Explicitly create a copy of the chunk
    chunk = only_target_offtarget.iloc[start:start+chunk_size].copy()
    
    # Use .loc for safe assignment
    chunk.loc[:, 'Embedding'] = chunk.apply(
        lambda row: ts.create_full_feature_vector(row['target'], row['offtarget_sequence']), axis=1
    )
    chunk.loc[:, 'Embedding'] = chunk['Embedding'].apply(lambda x: x.tolist())
    
    # Select only the required columns
    chunk = chunk[['Embedding', 'label']]
    
    # Append chunk to CSV
    chunk.to_csv(vectors_and_labels_csv, mode='a', header=False, index=False)
    
    # Free memory
    del chunk
    gc.collect()

print(f"Embedding process completed. Output saved to {vectors_and_labels_csv}")

# Save final_data_df
final_data_df.to_csv(final_data_csv, index=False)

# Zip the CSV files
zip_path = os.path.join(output_dir, 'data.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(vectors_and_labels_csv, os.path.basename(vectors_and_labels_csv))
    zipf.write(final_data_csv, os.path.basename(final_data_csv))

# Optionally remove the CSV files after zipping
os.remove(vectors_and_labels_csv)
os.remove(final_data_csv)

print(f"Zipped CSV files created at {zip_path}")
