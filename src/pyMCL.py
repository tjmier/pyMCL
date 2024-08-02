import numpy as np

def markov_clustering(
    matrix: np.ndarray,
    inflation: float = 2,
    expansion: float = 2,
    pruning_threshold: float = 1e-5,
    convergence_threshold: float = 0.001,
    keep_overlap: bool = False
) -> list:
    
    for i in (expansion, inflation):
        if i <= 1:
            msg = "Inflation and expansion values must be greater than 1"
            raise ValueError(msg)

    if matrix.shape[0] != matrix.shape[1]:
        msg = f"Input matrix must be a square matrix. Got shape {matrix.shape}"
        raise ValueError(msg)

    # Add self loops
    matrix = _add_self_loops(matrix)

    # MCL algorithm
    chaos = 1
    while True:
        # expansion
        matrix_exp = np.linalg.matrix_power(matrix, expansion)
        # purne
        matrix_prune = _prune(matrix_exp, pruning_threshold)
        # inflation
        matrix_inf = np.power(matrix_prune, inflation)
        # renormalize columns
        matrix = _normalize_cols(matrix_inf)
        # asess convergence
        chaos = _measure_convergence(matrix)
        if chaos < convergence_threshold:
            break
    
    # Last pruning
    matrix = _prune(matrix, pruning_threshold)

    clusters = _get_clusters(matrix)

    clusters = _resolve_overlapping_clusters(clusters, keep_overlap)

    # convert list of frozensets to list of sets
    clusters = [set(clust) for clust in clusters]

    return clusters

def _add_self_loops(matrix):
    
    # replace diagnol with 0
    np.fill_diagonal(matrix, 0.)
    #fill diagnol with max value in col
    np.fill_diagonal(matrix, np.max(matrix, axis=0))
    # normalzie cols
    matrix = _normalize_cols(matrix)
    
    return matrix

def _normalize_cols(matrix):
    row_sums = matrix.sum(axis=0, keepdims=True)
    return matrix / row_sums

def _prune(matrix, threshold):
    col_max = np.max(matrix, axis=0, keepdims=True)
    prune_mask = np.logical_and(matrix < threshold, matrix != col_max)
    matrix[prune_mask] = 0
    return matrix

def _measure_convergence(matrix):
    col_max = np.max(matrix, axis=0, keepdims=True)
    sum_of_squares = np.sum(np.square(matrix), axis=0)
    value = np.max(col_max - sum_of_squares)
    return value

def _get_clusters(matrix):

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

def _resolve_overlapping_clusters(clusters, keep_overlap):
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
        
        print("Overlapping clusters detected and automatically resolved.")
        return sorted(overlap_free_clusters, key=len, reverse=True)

    return sorted(clusters, key=len, reverse=True)


def clusters_to_adjacency(clusters):

    # convert list of sets to list of lists
    clusters = [list(clust) for clust in clusters]

    # number of nodes
    all_nodes = set(node for cluster in clusters for node in cluster)
    n_nodes = len(all_nodes)

    # Create an empty adjacency matrix
    matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    
    # Fill the matrix
    for cluster in clusters:
        matrix[np.ix_(cluster, cluster)] = 1
    
    return matrix

