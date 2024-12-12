import numpy as np
import pandas as pd
from generateTrainData.orginize_df import proccess_data
import generateTrainData.turn_sample_to_vector as ts

def test_functions():
    print("Starting function tests...\n")

    ### 1. Test `create_position_matrix`
    print("Testing `create_position_matrix`...")
    sgRNA = 'A'
    off_target = 'A'
    result = ts.create_position_matrix(sgRNA, off_target)
    print(f"  Input: sgRNA = {sgRNA}, off_target = {off_target}")
    print("  Expected diagonal matrix with 1 at (0,0).")
    print("  Result:\n", result, "\n")

    sgRNA = 'A'
    off_target = 'G'
    result = ts.create_position_matrix(sgRNA, off_target)
    print(f"  Input: sgRNA = {sgRNA}, off_target = {off_target}")
    print("  Expected matrix with 1 at (0,2).")
    print("  Result:\n", result, "\n")

    sgRNA = 'N'
    off_target = 'T'
    result = ts.create_position_matrix(sgRNA, off_target)
    print(f"  Input: sgRNA = {sgRNA}, off_target = {off_target}")
    print("  Expected matrix with all 1s (wildcard behavior).")
    print("  Result:\n", result, "\n")

    ### 2. Test `create_feature_vector`
    print("Testing `create_feature_vector`...")
    sgRNA_seq = "AAGTACGTACGTACGTACGTACG"  # Example sgRNA sequence
    off_target_seq = "AAGTTGCTACGTACGTTGCTACG"  # Example off-target sequence
    feature_vector = ts.create_feature_vector(sgRNA_seq, off_target_seq)
    print("  Input sequences:")
    print("    sgRNA_seq:", sgRNA_seq)
    print("    off_target_seq:", off_target_seq)
    print("  Expected feature vector of size 368.")
    print("  Result size:", len(feature_vector))
    print("  Feature vector (first 20 elements):", feature_vector[:20], "\n")

    ### 3. Test `compute_mismatch_count`
    print("Testing `compute_mismatch_count`...")
    mismatch_count = ts.compute_mismatch_count(sgRNA_seq, off_target_seq)
    print("  Input sequences:")
    print("    sgRNA_seq:", sgRNA_seq)
    print("    off_target_seq:", off_target_seq)
    print("  Expected mismatch count: 6 (example).")
    print("  Result:", mismatch_count, "\n")

    ### 4. Test `create_full_feature_vector`
    print("Testing `create_full_feature_vector`...")
    full_feature_vector = ts.create_full_feature_vector(sgRNA_seq, off_target_seq, include_mismatch_count=True)
    print("  Input sequences:")
    print("    sgRNA_seq:", sgRNA_seq)
    print("    off_target_seq:", off_target_seq)
    print("  Expected full feature vector of size 369 (368 + mismatch count).")
    print("  Result size:", len(full_feature_vector))
    print("  Full feature vector (first 20 elements):", full_feature_vector[:20], "\n")

    ### 5. Test `proccess_data` from `orginize_df`
    print("Testing `proccess_data`...")
    final_data_df, only_target_offtarget = proccess_data()
    print("  Result shapes:")
    print("    Final DataFrame shape:", final_data_df.shape)
    print("    Target/off-target DataFrame shape:", only_target_offtarget.shape)
    print("  Sample of final DataFrame:\n", final_data_df.head(), "\n")
    print("  Sample of target/off-target DataFrame:\n", only_target_offtarget.head(), "\n")

if __name__ == "__main__":
    test_functions()
