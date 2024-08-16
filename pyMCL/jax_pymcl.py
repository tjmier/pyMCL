"""Script for Markov Cluster Algorithm (MCL) implementation."""

# Built-in python modules
import warnings

# Third party modules
import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import lax


def jax_markov_cluster(
    matrix: np.ndarray,
    inflation: float = 2,
    expansion: float = 2,
    pruning_threshold: float = 1e-5,
    convergence_threshold: float = 0.001,
    overlaps: str = "resolve"
) -> list:
    """
    Perform Markov Cluster Algorithm (MCL) on a given numpy matrix matrix.

    Parameters
    ----------
    matrix : np.darray
        A square numpy matrix.
    inflation : float, optional, default=2
        The inflation parameter.
    expansion : float, optional, default=2
        The expansion parameter.
    pruning_threshold : float, optional, default=1e-5
        The pruning threshold.
    convergence_threshold : float, optional, default=1e-3
        The convergence threshold.
    overlaps : {"resolve", "keep"}, optional, default="resolve"
        Whether to resolve overlapping clusters in the final result
        or keep the the overlaps.

    Returns
    -------
    List
        List of sets containing nodes for each cluster.

    Raises
    ------
    ValueError
        If inflation or expansion is less than or equal to 1.
        If matrix is not a square matrix.

    """
    # check for pd dataframe
    is_dataframe = False
    if isinstance(matrix, pd.DataFrame):
        is_dataframe = True
        matrix, index_labels = _process_dataframe(matrix)

    for i in (expansion, inflation):
        if i <= 1:
            msg = "Inflation and expansion values must be greater than 1"
            raise ValueError(msg)

    if overlaps not in ["resolve", "keep"]:
        msg = "Overlaps must be either 'resolve' or 'keep'"
        raise ValueError(msg)

    if matrix.shape[0] != matrix.shape[1]:
        msg = f"Input matrix must be a square array. Got shape {matrix.shape}"
        raise ValueError(msg)

    # Add self loops
    matrix = _add_self_loops(matrix)

    # MCL algorithm
    chaos = 1
    while True:
        # expansion
        matrix_exp = jnp.linalg.matrix_power(matrix, expansion)
        # purne
        matrix_prune = _prune(matrix_exp, pruning_threshold)
        # inflation
        matrix_inf = jnp.power(matrix_prune, inflation)

        # renormalize columns
        matrix = _normalize_cols(matrix_inf)
        # asess convergence
        chaos = _measure_convergence(matrix)
        if chaos < convergence_threshold:
            break

    # Last pruning
    matrix = _prune(matrix, pruning_threshold)

    # Get clusters
    clusters = _get_clusters(matrix)
    clusters = _resolve_overlapping_clusters(clusters,
                                             keep_overlap = overlaps == "keep")

    # convert list of frozensets to list of sets
    clusters = [set(clust) for clust in clusters]

    # replace indices with intial labels
    if is_dataframe:
        clusters = [{index_labels[i] for i in clust} for clust in clusters]

    return clusters

def _process_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, list]:
    """
    Convert a pandas Dataframe to a numpy array and a list mapping the inital
    indices of the df.

    Parameters
    ----------
    df : pd.DataFrame
        A square pandas DataFrame.

    Returns
    -------
    np.ndarray
        The numpy array of the df.
    list
        The mapping of the initial indices of the df.

    """
    # reject 3x3 matrices or smaller
    min_shape = [4, 3]
    if df.shape[0] < min_shape[0]:
        msg = """Input dataframe too small. Must be at least 4x4 for NxN matices 
                or 4x3 for Nx3 matrices. Got shape {df.shape}"""
        raise ValueError(msg)

    # Pivot the DataFrame if it has 3 columns
    if df.shape[1] == 3:
        # Use iloc to select the first, second, and third columns
        df_pivot = df.pivot_table(index=df.columns[0],
                                  columns=df.columns[1],
                                  values=df.columns[2]).fillna(0)

        # Get all unique values from both the first and second columns
        col_0_values = set(df.iloc[:, 0])
        col_1_values = set(df.iloc[:, 1])
        all_values = col_0_values.union(col_1_values)

        # Reindex to include all unique values
        df = df_pivot.reindex(index=all_values, columns=all_values, fill_value=0)

    # check shape of the df
    elif df.shape[0] != df.shape[1]:
        msg = f"Input dataframe must have a shape of NxN or Nx3. Got shape {df.shape}"
        raise ValueError(msg)

    # dataframe columns and index
    df = df.sort_index(axis=0).sort_index(axis=1)
    columns = df.columns.tolist()
    indices = df.index.tolist()

    if columns.sort() != indices.sort():
        msg = """
            For dataframes of shape NxN, the index labels must match the column names.
            """
        raise ValueError(msg)

    # Convert the DataFrame to a numpy array
    matrix = df.to_numpy()

    return matrix, columns

def _add_self_loops(matrix: np.ndarray)->np.ndarray:
    # replace diagnol with 0
    np.fill_diagonal(matrix, 0.)
    #fill diagnol with max value in col
    np.fill_diagonal(matrix, np.max(matrix, axis=0))
    # normalzie cols
    matrix = _normalize_cols(matrix)

    return matrix

def _normalize_cols(matrix: np.ndarray)->np.ndarray:
    row_sums = matrix.sum(axis=0, keepdims=True)
    return matrix / row_sums

def _prune(matrix: jnp.ndarray, threshold: float) -> jnp.ndarray:
    col_max = jnp.max(matrix, axis=0, keepdims=True)
    # Create boolean mask
    prune_mask = jnp.logical_and(matrix < threshold, matrix != col_max)
    # Use `jax.lax.select` to apply the mask
    set_matrix = lax.select(prune_mask, jnp.zeros_like(matrix), matrix)
    return set_matrix

def _measure_convergence(matrix: np.ndarray)->float:
    col_max = np.max(matrix, axis=0, keepdims=True)
    sum_of_squares = np.sum(np.square(matrix), axis=0)
    value = np.max(col_max - sum_of_squares)
    return value

def _get_clusters(matrix: np.ndarray)->list:

    # Convert the matrix to a set of frozensets (clusters)
    clusters = set()
    for row in range(matrix.shape[0]):
        cluster = np.where(matrix[row, :] > 0)[0].tolist()
        clusters.add(frozenset(cluster))

    # Convert clusters to a sorted list by length for efficient subset checking
    sorted_clusters = sorted(clusters, key=len)

    # Identify subsets
    subsets = set()
    for i, clust1 in enumerate(sorted_clusters):
        for clust2 in sorted_clusters[i + 1:]:
            if clust1.issubset(clust2):
                subsets.add(clust1)
                break

    # Filter out the subsets
    filtered_clusters = [clust for clust in clusters if clust not in subsets]

    return filtered_clusters

def _resolve_overlapping_clusters(clusters: list, *, keep_overlap: bool)->list:
    # Check if there's an overlap in clusters
    all_nodes = set()
    overlap_detected = False
    for clust in clusters:
        if not clust.isdisjoint(all_nodes):
            overlap_detected = True
        all_nodes.update(clust)

    if overlap_detected and not keep_overlap:
        # sort clusters by length
        clusters = sorted(clusters, key=len, reverse=True)

        # Create a set to track nodes and clusters
        nodes_set = set()
        overlap_free_clusters = []
        for clust in clusters:
            # Add only those nodes that haven't been added to overlap_free_clusters
            new_clust = frozenset(node for node in clust if node not in nodes_set)
            if new_clust:
                overlap_free_clusters.append(new_clust)
                nodes_set.update(new_clust)

        msg = "Overlapping clusters detected and automatically resolved."
        warnings.warn(msg, stacklevel=2)
        return sorted(overlap_free_clusters, key=len, reverse=True)

    return sorted(clusters, key=len, reverse=True)


def jax_clusters_to_adjacency(clusters: list) -> pd.DataFrame | np.ndarray:
    """
    Convert a list of clusters into a corresponding adjacency matrix (a matrix where
    the array elements are 1 if the indices are in the same cluster and 0 otherwise).

    Parameters
    ----------
    clusters : list
        List of sets containing nodes for each cluster.

    Returns
    -------
    pd.DataFrame or np.ndarray
        The corresponding adjacency matrix for the list of clusters as a pandas
        DataFrame if the clusters contain strings, or as a numpy ndarray if the
        clusters contain only integers.

    """
    if not clusters:
        return np.ndarray([])

    # Check if clusters contain only integers
    all_elements = {elem for cluster in clusters for elem in cluster}
    is_integer = all(isinstance(elem, int) for elem in all_elements)

    # Convert list of sets to list of lists
    clusters = [list(clust) for clust in clusters]

    # Get all unique nodes
    all_nodes = sorted(all_elements)
    n_nodes = len(all_nodes)

    # Create an empty adjacency matrix
    matrix = np.zeros((n_nodes, n_nodes), dtype=int)

    # Fill the matrix
    for cluster in clusters:
        indices = [all_nodes.index(node) for node in cluster]
        matrix[np.ix_(indices, indices)] = 1

    if is_integer:
        return matrix

    # Create a DataFrame with the matrix and set the row and column labels
    return pd.DataFrame(matrix, index=all_nodes, columns=all_nodes)

