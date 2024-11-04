import pandas as pd
import zipfile
# Read the CSV file directly from the zip
with zipfile.ZipFile('cas_offinder_output2.zip') as z:
    with z.open('cas_offinder_output2.txt') as f:
        data_space = pd.read_csv(f, delimiter=r'\s+', header=None)
# create titles for each column
data_space.columns =['target', 'chrom', 'take_down', 'take_down2', 'take_down3', 'chromStart', 'offtarget_sequence', 'strand', 'distance', 'name']
# delete the unwanted colums
data_space.drop(columns=['take_down', 'take_down2', 'take_down3'], inplace=True)
# change the lowercase letters in the offtarget that represent the missmatches to uppercase
data_space['offtarget_sequence'] = data_space['offtarget_sequence'].str.upper()
changeseq_real_results =pd.read_excel('changeseq_final_results.xlsx')
print(data_space.head(1))
# print(data_space.shape)
print(changeseq_real_results.head(1))