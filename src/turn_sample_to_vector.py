import numpy as np

# Define nucleotide to index mapping
nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
# does not handle N :   
# def create_position_matrix(sgRNA, off_target):
#     # Initialize a 4x4 matrix for a single position
#     matrix = np.zeros((4, 4), dtype=int)
    
#     # Fill the matrix with 1s for matches and 0s for mismatches
#     sgRNA_index = nucleotide_map[sgRNA]
#     off_target_index = nucleotide_map[off_target]
    
#     if sgRNA_index == off_target_index:
#         # Match: Set the diagonal element to 1
#         matrix[sgRNA_index, off_target_index] = 1
#     else:
#         # Mismatch: Set the off-diagonal element to 1
#         matrix[sgRNA_index, off_target_index] = 1
    
#     return matrix

# does handle N- treats it as a wildcard-: 
def create_position_matrix(sgRNA, off_target):
    # Initialize a 4x4 matrix for a single position
    matrix = np.zeros((4, 4), dtype=int)
      
    # Convert nucleotides to their respective indices
    off_target_index = nucleotide_map[off_target]
    # handle cases when there is N in sgRNA, put match accourding to off target index
    if sgRNA =='N':
        
        matrix[off_target_index, off_target_index] = 1
        
    
    # # If either sgRNA or off_target is 'N', treat it as a match with any nucleotide
    # if sgRNA == 'N' or off_target == 'N':
    #     # Set all possible matches for the 'N' position
    #     matrix[:, :] = 1
    else: 
        # Convert nucleotides to their respective indices 
        sgRNA_index = nucleotide_map[sgRNA]      
        if sgRNA_index == off_target_index:
            # Match: Set the diagonal element to 1
            matrix[sgRNA_index, off_target_index] = 1
        else:
            # Mismatch: Set the off-diagonal element to 1
            matrix[sgRNA_index, off_target_index] = 1
    
    return matrix


def create_feature_vector(sgRNA_seq, off_target_seq):
    # Validate that the sequences have 23 positions
    # added validation for unwanted characters.
    assert len(sgRNA_seq) == 23 and len(off_target_seq) == 23, "Sequences must be 23 nucleotides long"
    assert all(base in nucleotide_map or base == 'N' for base in sgRNA_seq), "sgRNA contains invalid characters"
    assert all(base in nucleotide_map  for base in off_target_seq), "off_target contains invalid characters"
    
    # Create the 3D matrix (23 x 4 x 4)
    matrix_3d = np.array([create_position_matrix(sgRNA_seq[j], off_target_seq[j]) for j in range(23)])
    
    # Flatten to a 1D feature vector of size 368
    feature_vector = matrix_3d.flatten()
    
    return feature_vector

def compute_mismatch_count(sgRNA_seq, off_target_seq):
    # Count mismatches
    mismatches = sum(1 for s, o in zip(sgRNA_seq, off_target_seq) if s != o)
    return mismatches

def create_full_feature_vector(sgRNA_seq, off_target_seq, include_mismatch_count=False):
    # methods only handle upper case, so just incase input in not:
    sgRNA_seq = sgRNA_seq.upper()
    off_target_seq = off_target_seq.upper()
    # Create the 368-dimensional feature vector
    feature_vector = create_feature_vector(sgRNA_seq, off_target_seq)
    
    # Optionally add the mismatch count as the 369th feature
    if include_mismatch_count:
        mismatch_count = compute_mismatch_count(sgRNA_seq, off_target_seq)
        feature_vector = np.append(feature_vector, mismatch_count)
    
    return feature_vector

# # Example sequences
# sgRNA_seq = "ACGTACGTACGTACGTACGTACG"  # Example sgRNA sequence (23 nucleotides)
# off_target_seq = "ACGTTCGTACGTACGTTCGTACG"  # Example off-target sequence (23 nucleotides)

# # Generate the feature vector
# feature_vector = create_full_feature_vector(sgRNA_seq, off_target_seq, True)

# print("Feature vector size:", len(feature_vector))
# print("Feature vector:", feature_vector)
