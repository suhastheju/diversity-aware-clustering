def get_subset_index_to_id(subset_map):
    subset_index_to_id = {}
    index = 0
    for key in sorted(subset_map.keys()):
        subset_index_to_id[index] = key
        index += 1
    # end for
    return subset_index_to_id
