import pandas as pd
import zipfile

# Read the CSV file directly from the zip
with zipfile.ZipFile('cas_offinder_output2.zip') as z:
    with z.open('cas_offinder_output2.txt') as f:
        data_space = pd.read_csv(f, delimiter=r'\s+', header=None)  # Load data into a DataFrame with space as delimiter

# Create titles for each column in the DataFrame
data_space.columns = ['target', 'chrom', 'take_down', 'take_down2', 'take_down3', 'chromStart', 'offtarget_sequence', 'strand', 'distance', 'name']

# Delete the unwanted columns that are not needed for analysis
data_space.drop(columns=['take_down', 'take_down2', 'take_down3'], inplace=True)

# Change the lowercase letters in the offtarget sequence that represent mismatches to uppercase
data_space['offtarget_sequence'] = data_space['offtarget_sequence'].str.upper()

# Initialize the label column with value 0 for the first DataFrame
data_space['label'] = 0

# Read the real results from an Excel file into a DataFrame
changeseq_real_results = pd.read_excel('changeseq_final_results.xlsx')

# Clean the 'chrom' column by removing 'chr' prefix
changeseq_real_results['chrom'] = changeseq_real_results['chrom'].str.replace('chr', '', regex=False)

# Set the label column with value 1 for the second DataFrame
changeseq_real_results['label'] = 1

# Drop unwanted columns from the real results DataFrame
changeseq_real_results = changeseq_real_results.drop(columns=['chromEnd', 'CHANGEseq_reads', 'Unnamed: 7', 'chromStart:chromEnd'])

# Align the columns of data_space to match those of changeseq_real_results
data_space = data_space[changeseq_real_results.columns]

# Combine both DataFrames into one
combined_data = pd.concat([changeseq_real_results, data_space])

# Drop duplicate rows based on specific columns while keeping the first occurrence
final_data_df = combined_data.drop_duplicates(subset=['offtarget_sequence', 'name'], keep='first')

# Uncommented code that might be useful for debugging or exploration
# final_data_df = combined_data.drop_duplicates(subset=['name', 'chrom', 'chromStart'], keep='first')
# print(data_space.head())
# print(data_space.shape)
# print(changeseq_real_results.shape)
# print(changeseq_real_results.head())
# print(changeseq_real_results.columns)

# Print the sum of two specific numbers, possibly for verification of counts
print(840741 + 202043)

# Print the shape of the final DataFrame
print(final_data_df.shape)

# Print the first few rows of the final DataFrame
print(final_data_df.head())

# Count the number of rows in the final DataFrame where the label is 0
count = final_data_df[final_data_df['label'] == 0].shape[0]

# Print the count of rows with label 0
print(count)
