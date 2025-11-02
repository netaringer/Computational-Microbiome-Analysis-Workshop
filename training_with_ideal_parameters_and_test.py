import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.pipeline import Pipeline
from statsmodels.stats.multitest import multipletests
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from skbio.stats.composition import clr

def merge_data(pseudo_count=1.0):
    '''
    Merge microbiome, metabolome, and metadata into a single DataFrame and relative abundance for microbiome.
    Using CLR transformation for microbiome data.
    Args:
        pseudo_count: Small value to add to zeros before CLR (default: 0.1)
    Returns:
        merged_df: Combined DataFrame with microbiome, metabolome, and metadata['PATGROUPFINAL_C']
        num_microbiome: Number of microbiome features
        num_metabolom: Number of metabolome features   
    '''
    microbiome = pd.read_csv('microbiome.csv', index_col=0).fillna(0)
    microbiome_CLR = pd.DataFrame(clr(microbiome + pseudo_count), index=microbiome.index, columns=microbiome.columns)
    metabolom = pd.read_csv('serum_lipo.csv', index_col=0).fillna(0)
    metadata = pd.read_csv('metadata.csv', index_col=0).fillna(0)

    merged_df = pd.concat([microbiome_CLR, metabolom, metadata[['PATGROUPFINAL_C']]], axis=1)
    return merged_df, microbiome_CLR.shape[1] - 1, metabolom.shape[1] - 1



def variance_filtering(merged_df, microbiome_length, metabolome_length, microbiome_var_threshold=0.01, metabolome_var_threshold=1):
    """
    Preprocess microbiome and metabolome data with variance filtering.
    
    Args:
        merged_df: Combined dataframe with microbiome, metabolome, and metadata['PATGROUPFINAL_C']
        microbiome_var_threshold: Minimum variance threshold for microbiome features (default: 0.01)
        metabolome_var_threshold: Minimum variance threshold for metabolome features (default: 1)
    """
    # Get microbiome columns (first microbiome_length columns)
    biome_train = merged_df.iloc[:, :microbiome_length].copy()
    biome_variances = biome_train.var(axis=0)
    microbiome_filtered = biome_train.loc[:, biome_variances > microbiome_var_threshold]
    
    # Get metabolome columns
    met_train = merged_df.iloc[:, microbiome_length:(microbiome_length + metabolome_length)].copy()
    met_variances = met_train.var(axis=0)
    lipo_filtered = met_train.loc[:, met_variances > metabolome_var_threshold]
    
    # Combine filtered data with label column
    filtered_df = pd.concat([microbiome_filtered, lipo_filtered, merged_df[['PATGROUPFINAL_C']]], axis=1)
    
    print(f"Variance filtering: {microbiome_filtered.shape[1]}/{microbiome_length} microbiome, {lipo_filtered.shape[1]}/{metabolome_length} metabolome features kept")
    
    return filtered_df


def drop_highly_correlated(df, threshold=0.98):
    corr_matrix = df.corr().abs()  # Pearson correlation by default
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) # keeps only the upper triangle of the correlation matrix
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)] # keeps the second feature in case of high correlation
    return df.drop(columns=to_drop)

def filter_features(df, microbiome_length, metabolome_length, microbiome_var_threshold=3, metabolome_var_threshold=1, corr_threshold=0.95):
    filtered_by_variance = variance_filtering(df, microbiome_length, metabolome_length, microbiome_var_threshold, metabolome_var_threshold)
    
    feature_cols = [col for col in filtered_by_variance.columns if col not in ['SampleID', 'PATGROUPFINAL_C']]
    filtered_by_corr = drop_highly_correlated(filtered_by_variance[feature_cols], threshold=corr_threshold)

    filtered_df = pd.concat([filtered_by_corr, df[['PATGROUPFINAL_C']]], axis=1)

    print(f"Correlation filtering: {filtered_by_corr.shape[1]}/{len(feature_cols)} features kept after dropping highly correlated ones")
    return filtered_df

def create_matrix_and_vector(merged_df, quantiles=[0.33, 0.67]):
    """
    Create matrix and vector based on user-defined quantiles.
    
    Args:
        merged_df: Combined dataframe with microbiome, metabolome, and metadata
        microbiome_filtered: Filtered microbiome data
        lipo_filtered: Filtered metabolome data
        quantiles: List of quantile values (e.g., [0.25, 0.5, 0.75] for quartiles)
    
    Returns:
        feature_vector: Vector containing feature information
        matrix: Adjacency matrix
    """
    # Ensure quantiles are sorted
    quantiles = sorted(quantiles)
    n_ranges = len(quantiles) + 1  # Number of ranges = number of quantiles + 1
    
    n = 9 + (merged_df.shape[1] - 1) * n_ranges
    print(f"n = {n}, merged_df size = {merged_df.shape[1]}")
    print(f"Using quantiles: {quantiles}, creating {n_ranges} ranges")
    
    matrix = [[0 for i in range(n)] for j in range(n)]
    feature_vector = [0 for i in range(n)]

    labels = ['1', '2a', '2b', '3', '4', '5', '6', '7', '8']

    for i, label in enumerate(labels):
        feature_vector[i] = label

    i = 9
    for feature in merged_df.columns:
        if feature == 'SampleID' or feature == 'PATGROUPFINAL_C':   
            continue
        
        quantiles_ranges = []
        # Calculate quantile values for this feature
        for range_idx in range(n_ranges):
            if range_idx == 0:
                # First range: min to first quantile
                lower_bound = float(merged_df[feature].min())
                upper_bound = float(merged_df[feature].quantile(quantiles[0]))
                range_name = f'range_0_to_{quantiles[0]}'
            elif range_idx == n_ranges - 1:
                # Last range: last quantile to max
                lower_bound = float(merged_df[feature].quantile(quantiles[-1]))
                upper_bound = float(merged_df[feature].max())
                range_name = f'range_{quantiles[-1]}_to_1'
            else:
                # Middle ranges: between consecutive quantiles
                lower_bound = float(merged_df[feature].quantile(quantiles[range_idx - 1]))
                upper_bound = float(merged_df[feature].quantile(quantiles[range_idx]))
                range_name = f'range_{quantiles[range_idx-1]}_to_{quantiles[range_idx]}'

            feature_vector[i + range_idx] = (feature, range_name, (lower_bound, upper_bound))

            quantiles_ranges.append((lower_bound, upper_bound))
        
        for j, label in enumerate(labels):
            df = merged_df[merged_df['PATGROUPFINAL_C'] == label]

            feature_value = df[feature]
            range_idx = 0
            for lower_bound, upper_bound in quantiles_ranges:
                condition = (feature_value > lower_bound) & (feature_value <= upper_bound)

                # Calculate percentage for this range
                percentage = float(df[condition].count(axis=0).iloc[0] / df.shape[0])
                matrix[i + range_idx][j] = matrix[j][i + range_idx] = percentage
                
                range_idx += 1
        i += n_ranges

    return feature_vector, matrix

def find_significant_correlations(merged_df, matrix, alpha=0.05, quantiles=[0.33, 0.67]):

    correlations = []
    p_values = []
    n = len(quantiles) + 1
    
    # Build a list of feature columns (excluding SampleID and PATGROUPFINAL_C)
    feature_cols = [col for col in merged_df.columns if col not in ['SampleID', 'PATGROUPFINAL_C']]
    feature_to_idx = {feat: idx for idx, feat in enumerate(feature_cols)}

    for feat1 in feature_cols:
        for feat2 in feature_cols:
            if feat1 == feat2:   
                continue
            corr, p_val = spearmanr(merged_df[feat1], merged_df[feat2])
            correlations.append((feat1, feat2, corr))
            p_values.append(p_val)
    
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

    if len(reject) == 0:
        print(f"no correlations with p value <= {alpha} found")
        return [], matrix

    significant_correlations = [(feat1, feat2, corr) for (feat1, feat2, corr), rej in zip(correlations, reject) if rej]
    
    for feat1, feat2, corr in significant_correlations:
        # Use the feature index among feature columns only
        feat1_idx = feature_to_idx[feat1]
        feat2_idx = feature_to_idx[feat2]
        
        if corr > 0:
            for i in range(n):
                matrix_index1 = 9 + feat1_idx * n + i
                matrix_index2 = 9 + feat2_idx * n + i
                matrix[matrix_index1][matrix_index2] = corr
                matrix[matrix_index2][matrix_index1] = corr
        else:
            for i in range(n):
                matrix_index1 = 9 + feat1_idx * n + i
                matrix_index2 = 9 + feat2_idx * n + (n - 1 - i)
                matrix[matrix_index1][matrix_index2] = abs(corr)
                matrix[matrix_index2][matrix_index1] = abs(corr)

    return significant_correlations, matrix

def init_fluids_vectors(feature_vector):
    fluids_vectors = []
    for label in range(9):
        fluids_vector = [0 for i in range(len(feature_vector)) if i != label]
        fluids_vectors.append(fluids_vector)
    return fluids_vectors

def main_preprocessing(df, microbiome_length, metabolome_length, quantiles=[0.33, 0.67], microbiome_var_threshold=0.01, metabolome_var_threshold=1):

    filtered_df = filter_features(df, microbiome_length, metabolome_length, microbiome_var_threshold, metabolome_var_threshold)

    feature_vector, adjacency_matrix = create_matrix_and_vector(merged_df=filtered_df, quantiles=quantiles)

    significant_corr, adjacency_matrix = find_significant_correlations(merged_df=filtered_df, matrix=adjacency_matrix, alpha=0.05, quantiles=quantiles)

    for i in range(9):
        feature_vector[i] = 'Label_' + str(i+1)
    if len(significant_corr) == 0:
        print("this fold is useless")
        return None
    
    df = pd.DataFrame(adjacency_matrix, index=feature_vector, columns=feature_vector)
    df.to_csv("adjacency_matrix.csv")

    return df



########################FS_ALGO:########################

def cluster_match_partition(clusters, complement):
    """
    Check if clustering results match the expected group vs complement partition.
    
    Parameters:
    -----------
    clusters : list
        List of clusters, where each cluster is a list of labels
    group_to_check : set
        Set of labels representing the group to check
    complement : set
        Set of labels representing the complement group
        
    Returns:
    --------
    bool
        True if one of the clusters doesn't contain any complement labels (indicating a good partition),
        False otherwise
        
    Description:
    ------------
    This function checks whether the clustering results properly separate the group from its complement.
    If either cluster doesn't contain any complement labels, it means the clustering successfully
    separated the groups, so the function returns True.
    """
    if len(clusters) < 2:
        return False
    
    cluster1 = clusters[0]
    cluster2 = clusters[1]
    
    # Check if there is one cluster which is clear of any label that is a part of the complement
    # If either cluster1 or cluster2 contains NO complement labels, then we have a good separation
    cluster1_has_complement = any(label in cluster1 for label in complement)
    cluster2_has_complement = any(label in cluster2 for label in complement)
    
    # Return True if at least one cluster is completely clear of complement labels
    if not cluster1_has_complement or not cluster2_has_complement:
        return True
    else:
        return False


from scipy.spatial.distance import squareform
def compute_2_clusters_for_1_feature(label_value_dict):
    """
    Perform hierarchical clustering on labels based on their feature values to create 2 clusters.
    
    Parameters:
    -----------
    label_value_dict : dict
        Dictionary mapping labels to their feature values
        Format: {label: value}
        
    Returns:
    --------
    list
        List of two clusters, where each cluster is a sorted list of labels
        
    Description:
    ------------
    This function creates a distance matrix based on the absolute differences between
    feature values of different labels, then performs hierarchical clustering using
    average linkage to group the labels into exactly 2 clusters.
    """
    # Sort the labels
    labels = sorted(label_value_dict.keys())
    # Create a DataFrame for the distance matrix
    df = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for i in labels:
        for j in labels:
            df.loc[i, j] = abs(label_value_dict[i] - label_value_dict[j])

    # Convert the distance matrix to condensed form for linkage
    # (scipy expects a condensed distance matrix)
    condensed_dist = squareform(df.values)

    # Perform hierarchical clustering
    Z = linkage(condensed_dist, method='average')
    
    # Cut the dendrogram to form 2 clusters
    cluster_assignments = fcluster(Z, 2, criterion='maxclust')
    # Map labels to clusters
    label_to_cluster = dict(zip(labels, cluster_assignments))
    # Group labels by cluster
    clusters = defaultdict(list)
    for label, cluster_id in label_to_cluster.items():
        clusters[cluster_id].append(label)
    # Save the clusters for this feature
    clusters = [sorted(cluster) for cluster in clusters.values()]
    
    return clusters


def get_informative_feature_ranges(
    groups_to_check,
    feature_range_colors,
    single_group_threshold=0.00000000001
):
    """
    Identify informative feature-range nodes that can effectively separate groups.
    
    Parameters:
    groups_to_check : list
        List of tuples, where each tuple contains (group, complement) sets
        Example: [({1,2}, {3,4,5,6,7,8}), ({3,4,5,6}, {7,8}), ({7}, {8})]
        Each tuple represents a group and its complement to test feature separation
    feature_range_colors : dict
        Dictionary mapping feature names to label-value dictionaries
        Format: {feature_name: {label: value}}
    single_group_threshold : float, optional
        Minimum threshold for feature values to consider a node (default: 0.1)
        
    Returns: dict
        Dictionary mapping each group-complement tuple to a list of informative features
        Format: {(group, complement): [feature1, feature2, ...]}

    """


    selected_features_for_groups = {}


    for group_and_complement in groups_to_check:
        group_to_check = group_and_complement[0]
        complement = group_and_complement[1]
        # Convert sets to frozensets for use as dictionary keys
        group_complement_key = (frozenset(group_to_check), frozenset(complement))
        selected_features_for_groups[group_complement_key]=[]
        
        # Debug first iteration
        # debug_done = False
        
        for feature, label_value_dict in feature_range_colors.items():  
            
            threshold_passed = False
            for label in group_to_check | complement:
                if label not in label_value_dict:
                    print(f"WARNING: Label {label} not found in label_value_dict")
                    continue
                feature_val = label_value_dict[label]
                if feature_val >= single_group_threshold:
                    threshold_passed = True

            if threshold_passed:
                #check if clusters match partition
                cur_label_value_dict = {label: label_value_dict[label] for label in group_to_check | complement}
                # print("cur_label_value_dict:", cur_label_value_dict)
                feature_cluters = compute_2_clusters_for_1_feature(cur_label_value_dict)
                # print("feature:", feature)
                # print("group and complement:", group_to_check, complement)
                # print("feature cluters:", feature_cluters)
                match = cluster_match_partition(feature_cluters, complement)
                # print("match:", match)
                if match:
                    selected_features_for_groups[group_complement_key].append(feature)
        #print for each key, and the len of its value
    for k, v in selected_features_for_groups.items():
        print(f"Key: {k}, Number of selected features: {len(v)}")
    return selected_features_for_groups

def get_layers_as_dict(
    groups_to_check,
    feature_range_colors
):
    """
    Covers for 'get_informative_feature_ranges' function to return a dictionary without feature selection.
    
    Parameters:
    groups_to_check : list
        List of tuples, where each tuple contains (group, complement) sets
        Example: [({1,2}, {3,4,5,6,7,8}), ({3,4,5,6}, {7,8}), ({7}, {8})]
        Each tuple represents a group and its complement to test feature separation
    feature_range_colors : dict
        Dictionary mapping feature names to label-value dictionaries
        Format: {feature_name: {label: value}}
        
    Returns: dict
        Dictionary mapping each group-complement tuple to a list of all features

    """

    selected_features_for_groups = {}

    for group_and_complement in groups_to_check:
        group_to_check = group_and_complement[0]
        complement = group_and_complement[1]
        # Convert sets to frozensets for use as dictionary keys
        group_complement_key = (frozenset(group_to_check), frozenset(complement))
        selected_features_for_groups[group_complement_key]= feature_range_colors.keys()
        
    return selected_features_for_groups


import numpy as np
import scipy
import scipy.linalg
import scipy.sparse as sp
from multiprocessing import Pool, cpu_count, Manager, Lock
import pickle 
import pandas as pd
import math
import sys
import os
import json

# ------------------------ Propagation Functions ------------------------

def main_prop(mat, arr, eps=1e-6, alpha=0.1, 
              normalization_met=1, max_iter=1000):
    """
    Run network propagation on a gene network.
    Returns normalized propagation results and number of iterations.
    """
    p0_arr = sp.csr_matrix(arr).T
    n = p0_arr.shape[0]
    mat = sp.csr_matrix(mat)

    # Choose normalization
    if normalization_met == 1:
        W = get_normalized_mat(mat)
    else:
        W = get_normalized_mat_A_D_INV(mat)

    # Normalize with vector of ones
    for_norm_arr = sp.csr_matrix(np.ones(n)).T
    # for_norm_res, _ = get_prop_scores(W, for_norm_arr, eps, alpha, iter, min_iter, max_iter)

    # # Propagate on actual vector
    # res, iter_num = get_prop_scores(W, p0_arr, eps, alpha, iter, min_iter, max_iter)
    for_norm_res, _ = get_prop_scores(W, for_norm_arr, eps, alpha, max_iter)

    # Propagate on actual vector
    res, iter_num = get_prop_scores(W, p0_arr, eps, alpha, max_iter)

    

    return res / for_norm_res, iter_num


def get_prop_scores(mat, arr, eps=1e-6, alpha=0.1, max_iter=1000):
    """
    Network propagation with relative epsilon convergence.
    """
    iter_num = 0
    pk_1 = arr.copy()

    while iter_num < max_iter:
        pk = pk_1
        pk_1 = alpha * arr + (1 - alpha) * mat @ pk

        if math.sqrt(scipy.linalg.norm((pk_1 - pk).toarray())) < eps:
            break

        iter_num += 1
    
    return pk_1, iter_num



def get_normalized_mat(mat):
    """Symmetric normalization: W = D^(-1/2) * A * D^(-1/2)"""
    row_sums = np.array(mat.sum(axis=0)).flatten()
    row_sums[row_sums == 0] = 1
    D_inv_sqrt = sp.diags(1 / np.sqrt(row_sums))
    W = D_inv_sqrt * mat * D_inv_sqrt
    return W


def get_normalized_mat_A_D_INV(mat):
    """Random-walk normalization: W = A * D^(-1)"""
    mat = mat.toarray()
    D = np.diag(mat.sum(axis=1))
    D_inv = np.linalg.inv(D)
    W = mat @ D_inv
    return sp.csr_matrix(W)


def prop_wrapper(features_labels_network, eps=1e-6, alpha=0.1):
    """
    A wrapper that handles propagation calls on the input graph.

    Args:
        features_labels_network (pd.DataFrame): symmetric adjacency matrix with features and label nodes as index/columns
        eps (float): convergence tolerance for propagation
        alpha (float): damping factor for propagation

    Returns:
        pd.DataFrame: DataFrame with index = nodes, columns = labels (1-8), and values = propagation scores
    """

    num_labels = 9
    node_names = features_labels_network.index.tolist()

    output_graph = pd.DataFrame(
        np.zeros((features_labels_network.shape[0], num_labels)),
        index=node_names,
        columns=[f"Label_{i}" for i in range(1, num_labels + 1)]
    )

    csr_graph = sp.csr_matrix(features_labels_network.values)

    # 3. Find the indices of label nodes (assume their names are exactly 'Label_1' to 'Label_8')
    label_names = [f"Label_{i}" for i in range(1, num_labels + 1)]

    # 4. Propagate each label individually
    for label_name in label_names:
        if label_name not in node_names:
            continue  # skip if the label is not in the graph

        label_index = node_names.index(label_name)

        # Initialize the propagation vector p0 (column vector with 1 at label_index)
        p0 = np.zeros(features_labels_network.shape[0], dtype=float)
        p0[label_index] = 1.0

        # Call the propagation function
        res, iter_num = main_prop(csr_graph, p0, eps=eps, alpha=alpha)

        # Convert result to flat dense array if it's a sparse matrix
        if sp.issparse(res):
            res = res.toarray().flatten()
        else:
            res = np.asarray(res).flatten()

        # Store the result in the correct column of the output graph
        output_graph[label_name] = res

    return output_graph


########################### some of the WRAPPERs:############################


def build_feature_range_colors(output_graph):
# Keep only the feature rows (exclude the first 9 label rows)
    feature_rows = output_graph.drop(index=[f"Label_{i}" for i in range(1, 10)])

    # Build the dict
    feature_range_colors = {}

    for feature, row in feature_rows.iterrows():
        # Row contains values for Label_1 .. Label_9
        label_dict = {int(col.split("_")[1]): row[col] for col in output_graph.columns}
        # Use only first two parts of the tuple (feature_name, range_label)
        if isinstance(feature, tuple):
            feature_name = str(feature[0]) 
        else:
            feature_name = str(feature)
        feature_range_colors[feature_name] = label_dict

    return feature_range_colors
    

def wrapper_for_hirarchy_sub_group(hirarchy_sub_group):
    # Apply +1 to all integers
    updated_hirarchy_sub_group = [
        ( {x+1 for x in tup[0]}, {x+1 for x in tup[1]} ) 
        for tup in hirarchy_sub_group
    ]
    return updated_hirarchy_sub_group

###### hirarchial clustering algo #########
class Node:
    def __init__(self, idx, left=None, right=None, dist=0.0, leaves=None):
        self.idx = idx
        self.left = left
        self.right = right
        self.dist = float(dist)
        self.leaves = leaves if leaves is not None else set()

def build_tree_from_linkage(Z, n):
    """Builds a binary tree with leaf sets for each internal node."""
    nodes = {i: Node(i, None, None, 0.0, leaves={i}) for i in range(n)}
    for k, (a, b, dist, size) in enumerate(Z):
        a, b = int(a), int(b)
        idx = n + k
        left, right = nodes[a], nodes[b]
        nodes[idx] = Node(idx, left, right, dist=float(dist),
                          leaves=left.leaves | right.leaves)
    root = nodes[n + Z.shape[0] - 1]
    return root

def splits_towards_label(Z, target_label=8, labels=None):
    """
    Return list of bipartitions [(A,B), (A',B'), ...] where only the side
    containing `target_label` is further split (walk root -> leaf target).
    Each A,B are sets of external labels.
    """
    n = Z.shape[0] + 1
    if labels is None:
        labels = list(range(n))  # external labels corresponding to 0..n-1 indices

    # map internal node leaf indices -> external labels
    def to_label_set(indices_set):
        return {labels[i] for i in indices_set}

    root = build_tree_from_linkage(Z, n)
    splits = []
    node = root

    # descend until we reach the leaf that is 'target_label'
    while node.left is not None and node.right is not None:
        left_has_target  = (target_label in to_label_set(node.left.leaves))
        right_has_target = (target_label in to_label_set(node.right.leaves))

        if left_has_target and right_has_target:
            # Shouldn't happen in a proper binary tree, but guard anyway
            break

        # identify target child and the "other side"
        target_child = node.left if left_has_target else node.right
        other_child  = node.right if left_has_target else node.left

        # record the bipartition at this level
        splits.append((to_label_set(other_child.leaves),
                       to_label_set(target_child.leaves)))

        # continue down only the target side
        node = target_child

    return splits

def convert_propagation(df):
    #df = pd.read_csv(file, index_col=0)
    df = df.iloc[:9, :9]
    return df

def hirarchial_clustering(df):
    np.fill_diagonal(df.values, 1.0)

    # symmetrize + convert to distance
    sim = (df + df.T) / 2.0
    np.fill_diagonal(sim.values, 1.0)
    dist_sq = 1.0 - sim.values
    np.fill_diagonal(dist_sq, 0.0)
    dist_vec = squareform(dist_sq, checks=True)
    # hierarchical clustering (complete or average)
    Z = linkage(dist_vec, method="complete")

    # get splits walking only towards label 8
    path_splits = splits_towards_label(Z, target_label=8, labels=list(range(9)))
    # print(path_splits)
    return path_splits

######################################Creating Random Forest Models#############################################


def shift_dict_keys(d, shift=-1):

    new_dict = {}
    for key, value in d.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("Expected keys to be tuples of two frozensets")

        groupA, groupB = key
        newA = frozenset(x + shift for x in groupA)
        newB = frozenset(x + shift for x in groupB)
        new_dict[(newA, newB)] = value
    return new_dict

def clean_dict_values(d):
    
    def strip_after_last_underscore(s):
        if not isinstance(s, str):
            print(type(s))
            s = str(s)
        parts = s.rsplit("_", 1)
        return parts[0].strip() if len(parts) == 2 else s.strip()

    def unique_in_order(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {
        k: unique_in_order(s for s in v)
        for k, v in d.items()
    }


def table_filtering(groupA, groupB, features, df, label_col="PATGROUPFINAL_C"):
    '''creates filtered train df only with groupA (sick) as 0 and groupB (perhapes healthy) as 1 and only relevant features'''
    if df.index.name is not None:
        idx_name = str(df.index.name).strip()
    else:
        idx_name = None
        print("index.name is None")

    if idx_name is not None and any(str(f).strip() == idx_name for f in features):
        df = df.reset_index()

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    features = [str(f).strip() for f in features]
    label_col = str(label_col).strip()

    if label_col not in df.columns and df.index.name == label_col:
        df = df.reset_index()
        df.columns = [str(c).strip() for c in df.columns]

    labels_to_keep = list(groupA) + list(groupB)

    if label_col not in df.columns:
        raise KeyError(f"'{label_col}' not found in columns after normalization.")

    df = df.copy()
    df[label_col] = df[label_col].astype(str)
    labels_to_keep_str = set(str(x) for x in labels_to_keep)

    filtered_df = df[df[label_col].isin(labels_to_keep_str)].reset_index(drop=True)

    features_to_keep = features + [label_col]
    existing = [c for c in features_to_keep if c in filtered_df.columns]
    missing = [c for c in features_to_keep if c not in filtered_df.columns]

    if missing:
        print(f"{len(missing)} Missing features (dropped): {missing}")

    filtered_df = filtered_df.loc[:, existing].copy()

    mapping = {str(lbl): 0 for lbl in groupA}
    mapping.update({str(lbl): 1 for lbl in groupB})
    filtered_df[label_col] = filtered_df[label_col].map(mapping)

    filtered_df.to_csv(f"train_dataFrame_for_model_{groupA}_{groupB}.csv", index=False)
    return filtered_df

def build_random_forest(train, random_state=42, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    X_train = train.drop(columns=["PATGROUPFINAL_C"])
    y_train = train["PATGROUPFINAL_C"]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    return (model)


def build_list_of_RF(dict1, train_df, **rf_params):
    '''getting the dictionary from part 9, and hirarchy from part 8
    Accepts extra positional/keyword arguments to avoid "too many positional arguments" errors
    if this function is accidentally called with additional parameters.'''
    d_shifted = shift_dict_keys(dict1)
    d_cleaned = clean_dict_values(d_shifted)
    model_lst = []
    for key,value in d_cleaned.items():
        features = value
        groupA = key[0]
        groupB = key[1]
        df_for_model = table_filtering(groupA, groupB, features, train_df)
        model = build_random_forest(df_for_model, **rf_params)
        model_lst.append((model, features))

    return model_lst

def sample_evaluation(sample, models_lst, prob_thresholds):
    '''
    Evaluates probability-like ratio order for a given sample to be classified as sick
    
    Args:
        sample: A pandas Series representing a single sample
        models_lst: List of tuples (model, features)
        prob_thresholds: List of probability thresholds for each model

    Returns:
        probability for classification as sick
    '''
    probability = 1
    for i, (model, features) in enumerate(models_lst):
        # common_features = [f for f in features if f in sample.index]
        # vals = sample[common_features].values
        vals = sample[features].values

        X = pd.DataFrame([vals], columns=features)
        probs = model.predict_proba(X)[0]
        prob_sick = probs[0]
        # print("********************************************")
        # print(prob_sick)
        # print(prob_thresholds)
        # print("********************************************")

        probability *= prob_sick
        if prob_sick >= prob_thresholds[i]:
            return probability
    return probability

######################## REAL TEST #####################
def preprocess_real_test(pseudo_count=1.0):
    microbiome_test = pd.read_csv('microbiome_test.csv', index_col=0).fillna(0)
    microbiome_CLR_test = pd.DataFrame(clr(microbiome_test + pseudo_count), index=microbiome_test.index, columns=microbiome_test.columns)
    metabolom_test = pd.read_csv('serum_lipo_test.csv', index_col=0).fillna(0)
    merged_df_test = pd.concat([microbiome_CLR_test, metabolom_test], axis=1)
    return merged_df_test

def real_test(models_lst, test_df, prob_threshold=None):
    """
    Test the models on the provided DataFrame.
    1 - when predicted as sick (PATGROUPFINAL_C != 8)
    0 - when predicted as healthy (PATGROUPFINAL_C == 8)
    """
    sick_probs = []
    ids = []
    if prob_threshold is None:
        thresholds_list = [0.8]*(len(models_lst)-1) + [0.5]
    else:
        thresholds_list = [prob_threshold]*(len(models_lst)-1) + [0.5]
    for idx, row in test_df.iterrows():
        sick_prob = sample_evaluation(row, models_lst, thresholds_list)
        ids.append(idx)
        sick_probs.append(sick_prob)
    
    sick_probs_df = pd.DataFrame({'ID': ids, 'Probability': sick_probs})
    sick_probs_df.to_csv(f"Output.csv", index = False)

    return sick_probs_df


if __name__== "__main__":
    merged_df, num_microbiome_features, num_metabolom_features = merge_data()
    X = merged_df.drop(columns=["PATGROUPFINAL_C"])
    y = merged_df["PATGROUPFINAL_C"]

    adjacency_matrix = main_preprocessing(
    df = merged_df,
    microbiome_length=num_microbiome_features,
    metabolome_length=num_metabolom_features,
    quantiles = [0.33333, 0.66666],
    microbiome_var_threshold=6,
    metabolome_var_threshold=1)   
    propagation_graph = prop_wrapper(adjacency_matrix, alpha=0.1)
    feature_range_colors = build_feature_range_colors(propagation_graph)
    output_graph_converted = convert_propagation(propagation_graph)
    hirarchy_sub_group = hirarchial_clustering(output_graph_converted)
    updated_hirarchy_sub_group = wrapper_for_hirarchy_sub_group(hirarchy_sub_group)
    informative_feature_ranges = get_informative_feature_ranges(
        updated_hirarchy_sub_group,
        feature_range_colors,
        single_group_threshold=0.0000000001)

    final_models = build_list_of_RF(informative_feature_ranges, merged_df, n_estimators =100, max_depth = None, min_samples_split = 2, min_samples_leaf = 4) 

        ### REAL TEST ###
    models_lst = final_models
    test_df = preprocess_real_test()
    real_test(models_lst, test_df, 0.9)
