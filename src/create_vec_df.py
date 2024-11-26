import orginize_df
import turn_sample_to_vector as ts
import pandas as pd
final_data_df, only_target_offtarget = orginize_df.proccess_data()

# test:
# test_df = only_target_offtarget.head(5)
# vectors_and_labels_test = pd.DataFrame(test_df.apply(lambda row: ts.create_full_feature_vector(row['target'], row['offtarget_sequence']), axis=1), columns=['Embedding'])
# vectors_and_labels_test['label'] = only_target_offtarget.head(5)['label'].values
# print(vectors_and_labels_test.head(2))

# Create a new DataFrame with a single column by applying the function to each row
vectors_and_labels = pd.DataFrame(only_target_offtarget.apply(lambda row: ts.create_full_feature_vector(row['target'], row['offtarget_sequence']), axis=1), columns=['Embedding'])
# add the labels
vectors_and_labels['label'] = only_target_offtarget['label'].values
print(vectors_and_labels.shape)