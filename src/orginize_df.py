import pandas as pd
import zipfile

# Read the CSV file directly from the zip
def proccess_data():
    with zipfile.ZipFile('cas_offinder_output_3.zip') as z:
        with z.open('cas_offinder_output_3.txt') as f:
           data_space = pd.read_csv(f, delimiter=r'\s+', header=None, dtype={1: str})
  # Load data into a DataFrame with space as delimiter

    # Create titles for each column in the DataFrame
    data_space.columns = ['target', 'chrom', 'take_down', 'take_down2', 'take_down3', 'chromStart', 'offtarget_sequence', 'strand', 'distance', 'name']

    # Delete the unwanted columns that are not needed for analysis
    data_space.drop(columns=['take_down', 'take_down2', 'take_down3'], inplace=True)

    
    # Change the lowercase letters in the offtarget sequence that represent mismatches to uppercase
    data_space['offtarget_sequence'] = data_space['offtarget_sequence'].str.upper()
    # remove rows where 'offtarget_sequence' column  contains 'n' 
    data_space = data_space[~data_space['offtarget_sequence'].str.contains('N', na=False)]
    df = data_space['offtarget_sequence']

    # Initialize the label column with value 0 for the first DataFrame
    data_space['label'] = 0

    # Read the real results from an Excel file into a DataFrame
    changeseq_real_results = pd.read_excel('changeseq_final_results.xlsx') 
    # Remove rows where the 'offtarget_sequence' contains '-' or its length is not 23
    changeseq_real_results = changeseq_real_results[
    ~changeseq_real_results['offtarget_sequence'].str.contains('-', na=False) &
    (changeseq_real_results['offtarget_sequence'].str.len() == 23)
    ]
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
    final_data_df = combined_data.drop_duplicates(subset=['chromStart', 'name', 'chrom'], keep='first')
    only_target_offtarget = final_data_df[['target', 'offtarget_sequence', 'label']]
    print("#Finished organizing original data into data frames")
# CHECKING WHAT WAS REMOVED IN DUPLICATES:
#    # Add the index as a column to track original row indices
#     duplicates = combined_data[combined_data.duplicated(subset=['chromStart', 'name', 'chrom'], keep=False)].reset_index()

#     # Pair duplicates by merging them back with the original DataFrame
#     paired_duplicates = pd.merge(
#         duplicates,
#         duplicates,
#         on=['chromStart', 'name', 'chrom'],
#         suffixes=('_original', '_duplicate')
#     )

#     # Keep only rows where the original indices are not the same to avoid self-pairing
#     paired_duplicates = paired_duplicates[paired_duplicates['index_original'] != paired_duplicates['index_duplicate']]

#         # Save paired duplicates to a CSV file
#     paired_duplicates.to_csv('paired_duplicates2.csv', index=False)

#     print("Paired duplicates have been saved to 'paired_duplicates.csv'.")


# various test:  
    print(changeseq_real_results.shape)

    # Print the sum of two origianl dataframes
    # print(840741 + 202043)

    # Print the shape of the final DataFrame
    # print("final data df info:")
    # print(final_data_df.shape)

    # Print the first few rows of the final DataFrame
    # print(final_data_df.head())
    # print("only targrt and off target df info:")
    # print(only_target_offtarget.shape)
    # print(only_target_offtarget.head())

    # Count the number of rows in the final DataFrame where the label is 0
    # count = final_data_df[final_data_df['label'] == 0].shape[0]

    # # Print the count of rows with label 0
    # print(count)
    # print(202043 + count)

    return final_data_df, only_target_offtarget
#proccess_data()
