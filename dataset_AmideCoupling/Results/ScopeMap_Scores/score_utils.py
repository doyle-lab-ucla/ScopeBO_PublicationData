#################################################################################################
# This code was copied from (https://github.com/DeepSynthesis/ScopeMap/tree/main/SubstrateScore)
#################################################################################################


import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.preprocessing import StandardScaler


def entropy(data, k=5, distance_metric='euclidean'):
    """
    Estimate spatial entropy of numpy array using K-nearest neighbors method.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input numpy array containing spatial data points
    k : int, default=5
        Number of nearest neighbors to consider
    distance_metric : str, default='euclidean'
        Distance metric for KNN calculation, supports 'euclidean', 'manhattan', 'cosine', 'tanimoto'
        
    Returns:
    --------
    float
        Estimated spatial entropy value
    """
    if data.size == 0 or len(data) < k + 1:
        return 0.0
    
    if data.shape[1] == 0:
        raise ValueError("Data must contain numeric columns to calculate spatial entropy")
    
    n_points = len(data)
    
    if distance_metric == 'tanimoto':
        distance_matrix = squareform(pdist(data, metric='jaccard'))
        
        distances = []
        for i in range(n_points):
            dists = distance_matrix[i, :]
            dists = np.concatenate([dists[:i], dists[i+1:]])
            k_nearest = np.sort(dists)[:k]
            distances.append(k_nearest)
        distances = np.array(distances)
    else:
        nbrs = NearestNeighbors(n_neighbors=k+1, metric=distance_metric)
        nbrs.fit(data)
        distances, indices = nbrs.kneighbors(data)
        
        distances = distances[:, 1:]
    
    kth_distances = distances[:, -1]
    
    epsilon = 1e-10
    kth_distances = np.maximum(kth_distances, epsilon)
    
    dimension = data.shape[1]
    
    log_distances = np.log(kth_distances)
    
    entropy_estimate = (
        np.log(n_points - 1) - 
        np.log(k) + 
        dimension * np.log(2) + 
        dimension * np.mean(log_distances)
    )
    
    return entropy_estimate


def mean_squared_distance_to_nearest(complete_data, sampling_data, distance_metric='euclidean'):
    """
    Calculate the mean squared distance from each point in the complete dataset to its nearest sampling point.
    
    Parameters:
    -----------
    complete_data : numpy.ndarray
        Numpy array of the complete dataset
    sampling_data : numpy.ndarray
        Numpy array of sampling points
    distance_metric : str, default='euclidean'
        Distance metric for calculation
        
    Returns:
    --------
    float
        Mean squared distance from complete dataset points to nearest sampling points
    """
    if complete_data.size == 0 or sampling_data.size == 0:
        return 0.0
    
    if complete_data.shape[1] == 0 or sampling_data.shape[1] == 0:
        raise ValueError("Data must contain numeric columns to calculate distance")
    
    distance_matrix = cdist(complete_data, sampling_data, metric=distance_metric)
    
    min_distances = np.min(distance_matrix, axis=1)
    
    mean_squared_distance = np.mean(min_distances ** 2)
    
    return mean_squared_distance


def mean_distance_to_nearest(complete_data, sampling_data, distance_metric='euclidean'):
    """
    Calculate the mean distance from each point in the complete dataset to its nearest sampling point.
    
    Parameters:
    -----------
    complete_data : numpy.ndarray
        Numpy array of the complete dataset
    sampling_data : numpy.ndarray
        Numpy array of sampling points
    distance_metric : str, default='euclidean'
        Distance metric for calculation
        
    Returns:
    --------
    float
        Mean distance from complete dataset points to nearest sampling points
    """
    if complete_data.size == 0 or sampling_data.size == 0:
        return 0.0
    
    if complete_data.shape[1] == 0 or sampling_data.shape[1] == 0:
        raise ValueError("Data must contain numeric columns to calculate distance")
    
    distance_matrix = cdist(complete_data, sampling_data, metric=distance_metric)
    
    min_distances = np.min(distance_matrix, axis=1)
    
    mean_distance = np.mean(min_distances)
    
    return mean_distance


def calc_mean_distance(complete_space_df, sampling_points_df, not_feature_col, distance_metric='euclidean', scaler=None):
    """
    Calculate the mean distance from complete dataset to sampling points using standardization parameters from the complete space.
    
    Parameters:
    -----------
    complete_space_df : pandas.DataFrame
        DataFrame of the complete space, used to determine standardization parameters (if scaler is None)
    sampling_points_df : pandas.DataFrame
        DataFrame of sampling points
    not_feature_col : list
        List of non-feature columns to exclude
    distance_metric : str, default='euclidean'
        Distance metric for calculation
    scaler : sklearn.preprocessing.StandardScaler, optional
        Pre-trained standardizer. If provided, will be used directly; otherwise a new standardizer will be trained on complete_space_df
        
    Returns:
    --------
    float
        Mean distance value from complete dataset to sampling points
    """
    feature_cols = [col for col in complete_space_df.columns if col not in not_feature_col]
    
    complete_features = complete_space_df[feature_cols].select_dtypes(include=[np.number])
    sampling_features = sampling_points_df[feature_cols].select_dtypes(include=[np.number])
    
    if complete_features.empty or sampling_features.empty:
        raise ValueError("DataFrame must contain numeric feature columns to calculate distance")
    
    if scaler is None:
        common_cols = complete_features.columns.intersection(sampling_features.columns)
        complete_features = complete_features[common_cols]
        sampling_features = sampling_features[common_cols]
        
        scaler = StandardScaler()
        scaler.fit(complete_features)
    else:
        expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else sampling_features.columns
        complete_features = complete_features[expected_features]
        sampling_features = sampling_features[expected_features]
    
    normalized_complete = scaler.transform(complete_features)
    normalized_sampling = scaler.transform(sampling_features)
    
    return mean_distance_to_nearest(normalized_complete, normalized_sampling, distance_metric=distance_metric)


def calc_mean_squared_distance(complete_space_df, sampling_points_df, not_feature_col, distance_metric='euclidean', scaler=None):
    """
    Calculate the mean squared distance from complete dataset to sampling points using standardization parameters from the complete space.
    
    Parameters:
    -----------
    complete_space_df : pandas.DataFrame
        DataFrame of the complete space, used to determine standardization parameters (if scaler is None)
    sampling_points_df : pandas.DataFrame
        DataFrame of sampling points
    not_feature_col : list
        List of non-feature columns to exclude
    distance_metric : str, default='euclidean'
        Distance metric for calculation
    scaler : sklearn.preprocessing.StandardScaler, optional
        Pre-trained standardizer. If provided, will be used directly; otherwise a new standardizer will be trained on complete_space_df
        
    Returns:
    --------
    float
        Mean squared distance value from complete dataset to sampling points
    """
    feature_cols = [col for col in complete_space_df.columns if col not in not_feature_col]
    
    complete_features = complete_space_df[feature_cols].select_dtypes(include=[np.number])
    sampling_features = sampling_points_df[feature_cols].select_dtypes(include=[np.number])
    
    if complete_features.empty or sampling_features.empty:
        raise ValueError("DataFrame must contain numeric feature columns to calculate distance")
    
    if scaler is None:
        common_cols = complete_features.columns.intersection(sampling_features.columns)
        complete_features = complete_features[common_cols]
        sampling_features = sampling_features[common_cols]
        
        scaler = StandardScaler()
        scaler.fit(complete_features)
    else:
        expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else sampling_features.columns
        complete_features = complete_features[expected_features]
        sampling_features = sampling_features[expected_features]
    
    normalized_complete = scaler.transform(complete_features)
    normalized_sampling = scaler.transform(sampling_features)
    
    return mean_squared_distance_to_nearest(normalized_complete, normalized_sampling, distance_metric=distance_metric)


def calc_entropy(complete_space_df, sampling_points_df, not_feature_col, k=5, distance_metric='euclidean', scaler=None):
    """
    Calculate spatial entropy of sampling points using standardization parameters from the complete space.
    
    Parameters:
    -----------
    complete_space_df : pandas.DataFrame
        DataFrame of the complete space, used to determine standardization parameters (if scaler is None)
    sampling_points_df : pandas.DataFrame
        DataFrame of sampling points
    not_feature_col : list
        List of non-feature columns to exclude
    k : int, default=5
        Number of nearest neighbors to consider
    distance_metric : str, default='euclidean'
        Distance metric for KNN calculation
    scaler : sklearn.preprocessing.StandardScaler, optional
        Pre-trained standardizer. If provided, will be used directly; otherwise a new standardizer will be trained on complete_space_df
        
    Returns:
    --------
    float
        Estimated spatial entropy value of sampling points
    """
    feature_cols = [col for col in complete_space_df.columns if col not in not_feature_col]
    
    sampling_features = sampling_points_df[feature_cols].select_dtypes(include=[np.number])
    
    if sampling_features.empty:
        raise ValueError("DataFrame must contain numeric feature columns to calculate spatial entropy")
    
    if scaler is None:
        complete_features = complete_space_df[feature_cols].select_dtypes(include=[np.number])
        if complete_features.empty:
            raise ValueError("DataFrame must contain numeric feature columns to calculate spatial entropy")
        
        common_cols = complete_features.columns.intersection(sampling_features.columns)
        complete_features = complete_features[common_cols]
        sampling_features = sampling_features[common_cols]
        
        scaler = StandardScaler()
        scaler.fit(complete_features)
    else:
        expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else sampling_features.columns
        sampling_features = sampling_features[expected_features]
    
    normalized_sampling = scaler.transform(sampling_features)
    return entropy(normalized_sampling, k=k, distance_metric=distance_metric)
