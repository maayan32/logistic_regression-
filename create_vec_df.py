import orginize_df
import turn_sample_to_vector as ts
final_data_df = orginize_df.proccess_data()[0]
only_target_offtarget = orginize_df.proccess_data()[1]
# Create a new DataFrame with a single column by applying the function to each row
vectors_and_labels = only_target_offtarget.DataFrame(only_target_offtarget.apply(lambda row: ts.create_full_feature_vector(row['name'], row['offtarget_sequence']), axis=1), columns=['Embedding'])
# vectors_and_labels.shape