import pandas as pd
data_space =pd.read_csv('cas_offinder_output.txt',delimiter='\t', header=None)
data_space.columns =[]
changeseq_real_results =pd.read_excel('changeseq_final_results.xlsx')
print(data_space.head(1))
print(changeseq_real_results.head(1))