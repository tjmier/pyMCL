"""Tests for the pyMCL module."""

# Third-party modules
import numpy as np
import pandas as pd

# Testing module
from pyMCL import pymcl as mcl

# Example matrix and results from Dongen 2008 https://doi.org/10.1137/040608635
test_data_1 = np.array(
    [
        [0.2  , 0.25 , 0.   , 0.   , 0.   , 0.333, 0.25 , 0.   , 0.   , 0.25 , 0.   , 0.   ],
        [0.2  , 0.25 , 0.25 , 0.   , 0.2  , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
        [0.   , 0.25 , 0.25 , 0.2  , 0.2  , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
        [0.   , 0.   , 0.25 , 0.2  , 0.   , 0.   , 0.   , 0.2  , 0.2  , 0.   , 0.2  , 0.   ],
        [0.   , 0.25 , 0.25 , 0.   , 0.2  , 0.   , 0.25 , 0.2  , 0.   , 0.   , 0.   , 0.   ],
        [0.2  , 0.   , 0.   , 0.   , 0.   , 0.333, 0.   , 0.   , 0.   , 0.25 , 0.   , 0.   ],
        [0.2  , 0.   , 0.   , 0.   , 0.2  , 0.   , 0.25 , 0.   , 0.   , 0.25 , 0.   , 0.   ],
        [0.   , 0.   , 0.   , 0.2  , 0.2  , 0.   , 0.   , 0.2  , 0.2  , 0.   , 0.2  , 0.   ],
        [0.   , 0.   , 0.   , 0.2  , 0.   , 0.   , 0.   , 0.2  , 0.2  , 0.   , 0.2  , 0.333],
        [0.2  , 0.   , 0.   , 0.   , 0.   , 0.333, 0.25 , 0.   , 0.   , 0.25 , 0.   , 0.   ],
        [0.   , 0.   , 0.   , 0.2  , 0.   , 0.   , 0.   , 0.2  , 0.2  , 0.   , 0.2  , 0.333],
        [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.2  , 0.   , 0.2  , 0.333]
    ]
)

expected_clusters_1 = [{0, 5, 6, 9}, {1, 2, 4}, {3, 7, 8, 10, 11}]

def test_markov_cluster():
    clusters = mcl.markov_cluster(test_data_1)
    assert clusters == sorted(expected_clusters_1, key=len, reverse=True)

# Another example matrix and results from Dongen 2008 https://doi.org/10.1137/040608635
# This matrix results in cluster overlap
test_data_2 = np.array(
    [
        [0.5  , 0.333, 0.   , 0.   , 0.   , 0.   , 0.   ],
        [0.5  , 0.333, 0.333, 0.   , 0.   , 0.   , 0.   ],
        [0.   , 0.333, 0.333, 0.333, 0.   , 0.   , 0.   ],
        [0.   , 0.   , 0.333, 0.333, 0.333, 0.   , 0.   ],
        [0.   , 0.   , 0.   , 0.333, 0.333, 0.333, 0.   ],
        [0.   , 0.   , 0.   , 0.   , 0.333, 0.333, 0.5  ],
        [0.   , 0.   , 0.   , 0.   , 0.   , 0.333, 0.5  ]
    ]
)

expected_clusters_2a = [{3, 4, 5, 6}, {0, 1, 2, 3}]
expected_clusters_2b = [{3, 4, 5, 6}, {0, 1, 2}]

def test_markov_cluster_overlap():
    overlapping_clusters = mcl.markov_cluster(test_data_2, keep_overlap=True)
    assert overlapping_clusters == sorted(expected_clusters_2a, key=len, reverse=True)
    nooverlap_clusters = mcl.markov_cluster(test_data_2, keep_overlap=False)
    assert nooverlap_clusters == sorted(expected_clusters_2b, key=len, reverse=True)


def test_clusters_to_adjacency():
    clusters = [{0, 1, 2}, {3, 4}]
    expected_matrix = np.array([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1]
    ])
    result = mcl.clusters_to_adjacency(clusters)
    np.testing.assert_array_equal(result, expected_matrix)

def test_string_clusters():
    clusters_str = [{"cat", "dog", "bird"}, {"apple", "banana"}]
    data = {
        "apple": [1, 1, 0, 0, 0],
        "banana": [1, 1, 0, 0, 0],
        "bird": [0, 0, 1, 1, 1],
        "cat": [0, 0, 1, 1, 1],
        "dog": [0, 0, 1, 1, 1]
    }
    index = ["apple", "banana", "bird", "cat", "dog"]
    expected_matrix = pd.DataFrame(data, index=index)
    result = mcl.clusters_to_adjacency(clusters_str)
    np.testing.assert_array_equal(result, expected_matrix)

def test_empty_clusters():
    clusters = []
    expected_matrix = np.array([])
    result = mcl.clusters_to_adjacency(clusters)
    np.testing.assert_array_equal(result, expected_matrix)

def test_single_node_clusters():
    clusters = [{0}, {1}, {2}]
    expected_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    result = mcl.clusters_to_adjacency(clusters)
    np.testing.assert_array_equal(result, expected_matrix)

def test_overlapping_clusters():
    clusters = [{0, 1}, {1, 2}]
    expected_matrix = np.array([
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1]
    ])
    result = mcl.clusters_to_adjacency(clusters)
    np.testing.assert_array_equal(result, expected_matrix)
