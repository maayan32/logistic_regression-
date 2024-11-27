import orginize_df
import turn_sample_to_vector as ts
import pandas as pd
import zipfile
import os

# Process data
final_data_df, only_target_offtarget = orginize_df.proccess_data()
# For test:
only_target_offtarget = only_target_offtarget.head(5)
final_data_df = final_data_df.head(5)

# Create vectors and labels DataFrame
vectors_and_labels = pd.DataFrame(only_target_offtarget.apply(lambda row: ts.create_full_feature_vector(row['target'], row['offtarget_sequence']), axis=1), columns=['Embedding'])

# Convert numpy vectors to lists before saving to CSV
vectors_and_labels['Embedding'] = vectors_and_labels['Embedding'].apply(lambda x: x.tolist())

# Add labels
vectors_and_labels['label'] = only_target_offtarget['label'].values

print("vectors and labels shape:")
print(vectors_and_labels.shape)
print("#Finished embedding sequences into vectors")

# Output directory
output_dir = os.path.join(os.path.dirname(__file__), 'output_data')
os.makedirs(output_dir, exist_ok=True)

# Define the output CSV file paths with new names
vectors_and_labels_csv = os.path.join(output_dir, 'OTS_T_samples.csv')
final_data_csv = os.path.join(output_dir, 'OTS_T_info.csv')

# Save the DataFrames as CSV files with the new names
vectors_and_labels.to_csv(vectors_and_labels_csv, index=False)
final_data_df.to_csv(final_data_csv, index=False)

# Zip the CSV files
zip_path = os.path.join(output_dir, 'data.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(vectors_and_labels_csv, os.path.basename(vectors_and_labels_csv))
    zipf.write(final_data_csv, os.path.basename(final_data_csv))

# Optionally, remove the CSV files after zipping (if you want to keep only the zip)
os.remove(vectors_and_labels_csv)
os.remove(final_data_csv)

print(f"Zipped CSV files created at {zip_path}")
