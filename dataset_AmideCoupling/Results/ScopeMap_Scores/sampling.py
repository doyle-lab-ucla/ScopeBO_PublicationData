#################################################################################################
# This code was copied from (https://github.com/DeepSynthesis/ScopeMap/tree/main/SubstrateScore)
#################################################################################################


import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from scipy.stats.qmc import LatinHypercube, Sobol

def compute_cluster_centroid(cluster_points, metric='euclidean'):
    """
    Compute cluster centroid based on distance metric
    
    Parameters:
        cluster_points: numpy.ndarray, data points in the cluster
        metric: str, distance metric method
    
    Returns:
        numpy.ndarray, cluster centroid
    """
    if len(cluster_points) == 0:
        raise ValueError("Cluster point set cannot be empty")
    
    if metric == 'manhattan':
        return np.median(cluster_points, axis=0)
    else:
        return np.mean(cluster_points, axis=0)

def compute_pairwise_distances(X, Y=None, metric='euclidean'):
    """
    Compute pairwise distances between data points, supporting multiple distance metrics
    
    Parameters:
        X: numpy.ndarray, data point matrix
        Y: numpy.ndarray, optional, second group of data points. If None, compute internal distances of X
        metric: str, distance metric method, supports 'euclidean', 'manhattan', 'cosine'
    
    Returns:
        numpy.ndarray, distance matrix
    """
    if metric in ['euclidean', 'manhattan']:
        return pairwise_distances(X, Y, metric=metric)
    elif metric == 'cosine':
        return pairwise_distances(X, Y, metric='cosine')
    else:
        raise ValueError(f"Unsupported distance metric: {metric}. Supported metrics: 'euclidean', 'manhattan', 'cosine'")

def cvt_sampling_df_norepeat(data, k, not_feature_columns, max_iters=500, tol=1e-5, sampled_data=None, distance_metric='euclidean'):
    """
    CVT sampling algorithm (supports DataFrame input, preserves non-numeric columns, excludes previously sampled data)
    
    Parameters:
        data: DataFrame, containing feature columns and non-numeric columns (e.g., 'reactant_aldehyde', 'conv')
        k: number of center points to sample
        not_feature_columns: list, column names that don't participate in distance calculation (numeric type)
        max_iters: maximum number of iterations
        tol: convergence threshold for center point changes
        sampled_data: DataFrame, previously sampled data points (same format as data, with unreset index)
        distance_metric: str, distance metric method, supports 'euclidean', 'manhattan', 'cosine'
    
    Returns:
        centers: DataFrame, sampled center points (contains all original columns)
        unselected_points: DataFrame, unsampled points (all original columns)
    """
    if sampled_data is not None:
        sampled_keys = sampled_data[not_feature_columns].drop_duplicates()
        data_with_flag = data.copy()
        data_with_flag['_is_sampled'] = False
        
        for _, sampled_row in sampled_keys.iterrows():
            mask = True
            for col in not_feature_columns:
                mask = mask & (data_with_flag[col] == sampled_row[col])
            data_with_flag.loc[mask, '_is_sampled'] = True
        
        available_data = data_with_flag[~data_with_flag['_is_sampled']].drop('_is_sampled', axis=1).copy()
    else:
        available_data = data.copy()
    
    X = available_data.drop(not_feature_columns, axis=1).values
    
    if len(X) < k:
        raise ValueError(f"Available data point count ({len(X)}) is less than required sampling count ({k})")
    
    indices = np.random.choice(len(X), k, replace=False)
    centers_X = X[indices].copy()
    
    for iteration in range(max_iters):
        if np.isnan(centers_X).any():
            centers_X[np.isnan(centers_X)] = np.mean(centers_X[~np.isnan(centers_X)])
        distances = compute_pairwise_distances(X, centers_X, distance_metric)

        labels = np.argmin(distances, axis=1)
        
        new_centers_X = np.zeros_like(centers_X)
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centers_X[i] = compute_cluster_centroid(cluster_points, distance_metric)
            else:
                new_centers_X[i] = centers_X[i]
        
        if np.linalg.norm(new_centers_X - centers_X) < tol:
            print(f"CVT algorithm converged, iterations: {iteration}")
            break
        centers_X = new_centers_X
    
    if np.isnan(centers_X).any():
        centers_X[np.isnan(centers_X)] = np.mean(centers_X[~np.isnan(centers_X)])
    final_distances = compute_pairwise_distances(X, centers_X, distance_metric)
    
    selected_indices = []
    used_indices = set()
    
    for center_idx in range(k):
        distances_to_center = final_distances[:, center_idx]
        candidate_indices = np.argsort(distances_to_center)
        
        for candidate_idx in candidate_indices:
            if candidate_idx not in used_indices:
                selected_indices.append(candidate_idx)
                used_indices.add(candidate_idx)
                break
        else:
            raise ValueError(f"Cannot find enough non-duplicate sampling points. Need {k} points, but only found {len(selected_indices)} non-duplicate points")
    
    selected_indices = np.array(selected_indices)
    
    centers = available_data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), centers.index)
    unselected_points = data.iloc[unselected_indices].copy()
    return centers, unselected_points
def fps_sampling_df_norepeat(data, k, not_feature_columns, sampled_data=None, random_state=None, distance_metric='euclidean'):
    """
    FPS (Farthest Point Sampling) algorithm (supports DataFrame input, preserves non-numeric columns, excludes previously sampled data)
    
    Parameters:
        data: DataFrame, containing feature columns and non-numeric columns (e.g., 'reactant_aldehyde', 'conv')
        k: number of center points to sample
        not_feature_columns: list, column names that don't participate in distance calculation (numeric type)
        sampled_data: DataFrame, previously sampled data points (same format as data, with unreset index)
        random_state: int, random seed
        distance_metric: str, distance metric method, supports 'euclidean', 'manhattan', 'cosine'
    
    Returns:
        centers: DataFrame, sampled center points (contains all original columns)
        unselected_points: DataFrame, unsampled points (all original columns)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if sampled_data is not None:
        sampled_keys = sampled_data[not_feature_columns].drop_duplicates()
        data_with_flag = data.copy()
        data_with_flag['_is_sampled'] = False
        
        for _, sampled_row in sampled_keys.iterrows():
            mask = True
            for col in not_feature_columns:
                mask = mask & (data_with_flag[col] == sampled_row[col])
            data_with_flag.loc[mask, '_is_sampled'] = True
        
        available_data = data_with_flag[~data_with_flag['_is_sampled']].drop('_is_sampled', axis=1).copy()
    else:
        available_data = data.copy()
    
    X = available_data.drop(not_feature_columns, axis=1).values
    
    if len(X) < k:
        raise ValueError(f"Available data point count ({len(X)}) is less than required sampling count ({k})")
    
    selected_indices = []
    
    first_idx = np.random.randint(0, len(X))
    selected_indices.append(first_idx)
    
    distance_matrix = compute_pairwise_distances(X, X, distance_metric)
    
    for i in range(k - 1):
        min_distances = np.full(len(X), np.inf)
        
        for selected_idx in selected_indices:
            distances_to_selected = distance_matrix[:, selected_idx]
            min_distances = np.minimum(min_distances, distances_to_selected)
        
        min_distances[selected_indices] = 0
        
        farthest_idx = np.argmax(min_distances)
        selected_indices.append(farthest_idx)
    
    selected_indices = np.array(selected_indices)
    
    centers = available_data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), centers.index)
    unselected_points = data.iloc[unselected_indices].copy()
    
    return centers, unselected_points

def lhs_sampling_df_norepeat(data, k, not_feature_columns, sampled_data=None, distance_metric='euclidean'):
    """
    Use Latin Hypercube Sampling (LHS) to select sample points from DataFrame
    
    Parameters:
    data: pd.DataFrame - DataFrame containing all data
    k: int - number of center points to sample
    not_feature_columns: list - list of column names that are not used as features
    sampled_data: DataFrame, previously sampled data points (same format as data, with unreset index)
    distance_metric: str, distance metric method, supports 'euclidean', 'manhattan', 'cosine'
    
    Returns:
    centers: pd.DataFrame - sampled center points (contains all original columns)
    unselected_points: pd.DataFrame - unsampled points (contains all original columns)
    """
    if sampled_data is not None:
        sampled_keys = sampled_data[not_feature_columns].drop_duplicates()
        data_with_flag = data.copy()
        data_with_flag['_is_sampled'] = False
        
        for _, sampled_row in sampled_keys.iterrows():
            mask = True
            for col in not_feature_columns:
                mask = mask & (data_with_flag[col] == sampled_row[col])
            data_with_flag.loc[mask, '_is_sampled'] = True
        
        available_data = data_with_flag[~data_with_flag['_is_sampled']].drop('_is_sampled', axis=1).copy()
    else:
        available_data = data.copy()
    
    X = available_data.drop(not_feature_columns, axis=1).values
    
    if len(X) < k:
        raise ValueError(f"Available data point count ({len(X)}) is less than required sampling count ({k})")
    
    sampler = LatinHypercube(d=X.shape[1])
    samples = sampler.random(n=k)
    
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    lhs_points = samples * (max_vals - min_vals) + min_vals
    
    if np.isnan(lhs_points).any():
        lhs_points[np.isnan(lhs_points)] = np.mean(lhs_points[~np.isnan(lhs_points)])
    final_distances = compute_pairwise_distances(X, lhs_points, distance_metric)
    
    selected_indices = []
    used_indices = set()
    
    for lhs_idx in range(k):
        distances_to_lhs = final_distances[:, lhs_idx]
        candidate_indices = np.argsort(distances_to_lhs)
        
        for candidate_idx in candidate_indices:
            if candidate_idx not in used_indices:
                selected_indices.append(candidate_idx)
                used_indices.add(candidate_idx)
                break
        else:
            raise ValueError(f"Cannot find enough non-duplicate sampling points. Need {k} points, but only found {len(selected_indices)} non-duplicate points")
    
    selected_indices = np.array(selected_indices)
    
    centers = available_data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), centers.index)
    unselected_points = data.iloc[unselected_indices].copy()
    return centers, unselected_points

def rand_sampling_df_no_repeat(data, k, not_feature_columns, sampled_data=None, distance_metric='euclidean'):
    """
    Use random sampling to select sample points from DataFrame
    
    Parameters:
    data: pd.DataFrame - DataFrame containing all data
    k: int - number of center points to sample
    not_feature_columns: list - list of column names that are not used as features
    sampled_data: DataFrame, previously sampled data points (same format as data, with unreset index)
    distance_metric: str, distance metric method, supports 'euclidean', 'manhattan', 'cosine'
    
    Returns:
    centers: pd.DataFrame - sampled center points (contains all original columns)
    unselected_points: pd.DataFrame - unsampled points (contains all original columns)
    """
    if sampled_data is not None:
        sampled_keys = sampled_data[not_feature_columns].drop_duplicates()
        data_with_flag = data.copy()
        data_with_flag['_is_sampled'] = False
        
        for _, sampled_row in sampled_keys.iterrows():
            mask = True
            for col in not_feature_columns:
                mask = mask & (data_with_flag[col] == sampled_row[col])
            data_with_flag.loc[mask, '_is_sampled'] = True
        
        available_data = data_with_flag[~data_with_flag['_is_sampled']].drop('_is_sampled', axis=1).copy()
    else:
        available_data = data.copy()

    feature_columns = [col for col in available_data.columns if col not in not_feature_columns]
    
    if not all(pd.api.types.is_numeric_dtype(available_data[col]) for col in feature_columns):
        raise ValueError("All feature columns must be numeric")
    
    if k > len(available_data):
        raise ValueError(f"Sampling count {k} cannot be greater than dataset size {len(data)}")
    
    center_indices = np.random.choice(len(available_data), k, replace=False)
    
    centers = available_data.iloc[center_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), center_indices)
    unselected_points = data.iloc[unselected_indices].copy()
    
    return centers, unselected_points

def classify_by_centers(data, centers, not_feature_columns):
    """
    Get classification for a group of data
    
    Parameters:
        data: DataFrame, containing feature columns and non-numeric columns (e.g., 'reactant_aldehyde', 'conv')
        centers: DataFrame, containing feature columns and non-numeric columns (e.g., 'reactant_aldehyde', 'conv')
        not_feature_columns: list, column names of non-descriptive columns
    
    Returns:
        output: DataFrame, with added "labels" classification column (contains all original columns)
        Note: categories correspond to centers' indices
    """
    raw_data = data.copy()
    for not_feature_column in not_feature_columns:
        if not_feature_column in data.columns:
            data = data.drop(not_feature_column, axis=1)
        if not_feature_column in centers.columns:
            centers = centers.drop(not_feature_column, axis=1)
    data = data.values
    centers = centers.values
    distances = np.sqrt(((data[:, np.newaxis, :] - centers) ** 2).sum(axis=2))
    
    labels = np.argmin(distances, axis=1).astype(int)
    if 'labels' in raw_data.columns:
        raw_data = raw_data.drop('labels', axis=1)
    output_data = pd.concat([raw_data, pd.DataFrame(labels, columns=['labels'])], axis=1)

    
    return output_data

def sobol_sampling_df_norepeat(data, k, not_feature_columns, sampled_data=None, distance_metric='euclidean'):
    """
    Use Sobol sequence sampling to select sample points from DataFrame
    
    Parameters:
    data: pd.DataFrame - DataFrame containing all data
    k: int - number of center points to sample
    not_feature_columns: list - list of column names that are not used as features
    sampled_data: DataFrame, previously sampled data points (same format as data, with unreset index)
    distance_metric: str, distance metric method, supports 'euclidean', 'manhattan', 'cosine'
    
    Returns:
    centers: pd.DataFrame - sampled center points (contains all original columns)
    unselected_points: pd.DataFrame - unsampled points (contains all original columns)
    """
    if sampled_data is not None:
        sampled_keys = sampled_data[not_feature_columns].drop_duplicates()
        data_with_flag = data.copy()
        data_with_flag['_is_sampled'] = False
        
        for _, sampled_row in sampled_keys.iterrows():
            mask = True
            for col in not_feature_columns:
                mask = mask & (data_with_flag[col] == sampled_row[col])
            data_with_flag.loc[mask, '_is_sampled'] = True
        
        available_data = data_with_flag[~data_with_flag['_is_sampled']].drop('_is_sampled', axis=1).copy()
    else:
        available_data = data.copy()
    
    X = available_data.drop(not_feature_columns, axis=1).values
    
    if len(X) < k:
        raise ValueError(f"Available data point count ({len(X)}) is less than required sampling count ({k})")
    
    sampler = Sobol(d=X.shape[1])
    samples = sampler.random(n=k)
    
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    sobol_points = samples * (max_vals - min_vals) + min_vals
    
    if np.isnan(sobol_points).any():
        sobol_points[np.isnan(sobol_points)] = np.mean(sobol_points[~np.isnan(sobol_points)])
    final_distances = compute_pairwise_distances(X, sobol_points, distance_metric)
    
    selected_indices = []
    used_indices = set()
    
    for sobol_idx in range(k):
        distances_to_sobol = final_distances[:, sobol_idx]
        candidate_indices = np.argsort(distances_to_sobol)
        
        for candidate_idx in candidate_indices:
            if candidate_idx not in used_indices:
                selected_indices.append(candidate_idx)
                used_indices.add(candidate_idx)
                break
        else:
            raise ValueError(f"Cannot find enough non-duplicate sampling points. Need {k} points, but only found {len(selected_indices)} non-duplicate points")
    
    selected_indices = np.array(selected_indices)
    
    centers = available_data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), centers.index)
    unselected_points = data.iloc[unselected_indices].copy()
    return centers, unselected_points


def kennard_stone_sampling_df_norepeat(data, k, not_feature_columns, sampled_data=None, distance_metric='euclidean', random_state=None):
    """
    Kennard-Stone sampling algorithm (supports DataFrame input, preserves non-numeric columns, excludes previously sampled data)
    
    Kennard-Stone algorithm is a distance-based sampling method that ensures sample representativeness by selecting points that are farthest from each other in the sample space.
    Algorithm steps:
    1. Select the two points with the maximum distance as initial points
    2. For each remaining point, select the point that is farthest from the already selected point set
    
    Parameters:
        data: DataFrame, containing feature columns and non-numeric columns (e.g., 'reactant_aldehyde', 'conv')
        k: number of center points to sample
        not_feature_columns: list, column names that don't participate in distance calculation (numeric type)
        sampled_data: DataFrame, previously sampled data points (same format as data, with unreset index)
        distance_metric: str, distance metric method, supports 'euclidean', 'manhattan', 'cosine'
        random_state: int, random seed, used for random selection when distances are equal
    
    Returns:
        centers: DataFrame, sampled center points (contains all original columns)
        unselected_points: DataFrame, unsampled points (all original columns)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if sampled_data is not None:
        sampled_keys = sampled_data[not_feature_columns].drop_duplicates()
        data_with_flag = data.copy()
        data_with_flag['_is_sampled'] = False
        
        for _, sampled_row in sampled_keys.iterrows():
            mask = True
            for col in not_feature_columns:
                mask = mask & (data_with_flag[col] == sampled_row[col])
            data_with_flag.loc[mask, '_is_sampled'] = True
        
        available_data = data_with_flag[~data_with_flag['_is_sampled']].drop('_is_sampled', axis=1).copy()
    else:
        available_data = data.copy()
    
    X = available_data.drop(not_feature_columns, axis=1).values
    
    if len(X) < k:
        raise ValueError(f"Available data point count ({len(X)}) is less than required sampling count ({k})")
    
    if k < 2:
        raise ValueError("Kennard-Stone sampling requires at least 2 sampling points")
    
    distance_matrix = compute_pairwise_distances(X, X, distance_metric)
    
    selected_indices = []
    remaining_indices = list(range(len(X)))
    
    max_distance = 0
    initial_pair = (0, 1)
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if distance_matrix[i, j] > max_distance:
                max_distance = distance_matrix[i, j]
                initial_pair = (i, j)
    
    selected_indices.extend([initial_pair[0], initial_pair[1]])
    remaining_indices.remove(initial_pair[0])
    remaining_indices.remove(initial_pair[1])
    
    for _ in range(k - 2):
        max_min_distance = -1
        best_candidate = None
        
        for candidate_idx in remaining_indices:
            min_distance_to_selected = float('inf')
            
            for selected_idx in selected_indices:
                distance = distance_matrix[candidate_idx, selected_idx]
                min_distance_to_selected = min(min_distance_to_selected, distance)
            
            if min_distance_to_selected > max_min_distance:
                max_min_distance = min_distance_to_selected
                best_candidate = candidate_idx
            elif min_distance_to_selected == max_min_distance and best_candidate is not None:
                if np.random.random() < 0.5:
                    best_candidate = candidate_idx
        
        if best_candidate is not None:
            selected_indices.append(best_candidate)
            remaining_indices.remove(best_candidate)
        else:
            if remaining_indices:
                random_choice = np.random.choice(remaining_indices)
                selected_indices.append(random_choice)
                remaining_indices.remove(random_choice)
    
    selected_indices = np.array(selected_indices)
    
    centers = available_data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), centers.index)
    unselected_points = data.iloc[unselected_indices].copy()
    
    print(f"Kennard-Stone sampling completed, sampled {len(centers)} points")
    return centers, unselected_points

def ward_clustering_df_norepeat(data, k, not_feature_columns, sampled_data=None, distance_metric='euclidean'):
    """
    Ward hierarchical clustering sampling algorithm (supports DataFrame input, preserves non-numeric columns, excludes previously sampled data)
    Uses Ward connectivity criterion for agglomerative hierarchical clustering, selects center point of each cluster as representative
    
    Parameters:
        data: DataFrame, containing feature columns and non-numeric columns (e.g., 'reactant_aldehyde', 'conv')
        k: number of center points to sample
        not_feature_columns: list, column names that don't participate in distance calculation (numeric type)
        sampled_data: DataFrame, previously sampled data points (same format as data, with unreset index)
        distance_metric: str, distance metric method, supports 'euclidean', 'manhattan', 'cosine'
    
    Returns:
        centers: DataFrame, sampled center points (contains all original columns)
        unselected_points: DataFrame, unsampled points (all original columns)
    """
    if sampled_data is not None:
        sampled_keys = sampled_data[not_feature_columns].drop_duplicates()
        data_with_flag = data.copy()
        data_with_flag['_is_sampled'] = False
        
        for _, sampled_row in sampled_keys.iterrows():
            mask = True
            for col in not_feature_columns:
                mask = mask & (data_with_flag[col] == sampled_row[col])
            data_with_flag.loc[mask, '_is_sampled'] = True
        
        available_data = data_with_flag[~data_with_flag['_is_sampled']].drop('_is_sampled', axis=1).copy()
    else:
        available_data = data.copy()
    
    X = available_data.drop(not_feature_columns, axis=1).values
    
    if len(X) < k:
        raise ValueError(f"Available data point count ({len(X)}) is less than required sampling count ({k})")
    
    if distance_metric == 'euclidean':
        linkage = 'ward'
        metric = 'euclidean'
    elif distance_metric == 'manhattan':
        linkage = 'complete'
        metric = 'manhattan'
    elif distance_metric == 'cosine':
        linkage = 'complete'
        metric = 'cosine'
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}. Supported metrics: 'euclidean', 'manhattan', 'cosine'")
    
    print(f"Using {linkage} linkage and {metric} distance metric for hierarchical clustering")
    
    clustering = AgglomerativeClustering(
        n_clusters=k,
        linkage=linkage,
        metric=metric
    )
    
    if np.isnan(X).any():
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    
    cluster_labels = clustering.fit_predict(X)
    
    selected_indices = []
    used_indices = set()
    
    for cluster_id in range(k):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        cluster_points = X[cluster_indices]
        
        cluster_center = np.mean(cluster_points, axis=0)
        
        distances_to_center = compute_pairwise_distances(
            cluster_points, cluster_center.reshape(1, -1), distance_metric
        ).flatten()
        
        sorted_cluster_indices = cluster_indices[np.argsort(distances_to_center)]
        
        for candidate_idx in sorted_cluster_indices:
            if candidate_idx not in used_indices:
                selected_indices.append(candidate_idx)
                used_indices.add(candidate_idx)
                break
        else:
            best_idx = sorted_cluster_indices[0]
            selected_indices.append(best_idx)
    
    if len(selected_indices) < k:
        remaining_indices = np.setdiff1d(np.arange(len(X)), selected_indices)
        additional_needed = k - len(selected_indices)
        if len(remaining_indices) >= additional_needed:
            additional_indices = np.random.choice(remaining_indices, additional_needed, replace=False)
            selected_indices.extend(additional_indices)
        else:
            additional_indices = np.random.choice(len(X), additional_needed, replace=True)
            selected_indices.extend(additional_indices)
    
    selected_indices = np.array(selected_indices[:k])
    
    centers = available_data.iloc[selected_indices].copy()
    unselected_indices = np.setdiff1d(np.arange(len(data)), centers.index)
    unselected_points = data.iloc[unselected_indices].copy()
    
    print(f"Ward clustering sampling completed, sampled {len(centers)} points")
    return centers, unselected_points
