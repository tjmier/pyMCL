"""Tests for the pyMCL module."""

# Third-party modules
import numpy as np
import pandas as pd
import pytest

# Testing module
from pyMCL import pymcl as mcl

# Example matrices and results from Dongen 2008 https://doi.org/10.1137/040608635
test1_array = np.array(
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

# This array results in overlapping clusters
test2_array = np.array(
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

test1_clusters = [
    {0, 5, 6, 9},
    {1, 2, 4},
    {3, 7, 8, 10, 11}
]

test2_clusters = [
    {3, 4, 5, 6},
    {0, 1, 2}
]

@pytest.mark.parametrize(
    ("input_data", "expected"),
    [
        (test1_array, test1_clusters),
        (test2_array, test2_clusters),
    ]
)

def test_mcl_array_input(input_data, expected):
    result = mcl.markov_cluster(input_data)
    assert result == sorted(expected, key=len, reverse=True)

def convert_to_df(array):
    # convert to dataframe and asiign col and row labels
    labels = [chr(i) for i in range(ord("a"), ord("a") + array.shape[0])]
    as_df = pd.DataFrame(array, index=labels, columns=labels)

    # shuffle rows and columns
    rng = np.random.default_rng()
    row_perm = rng.permutation(as_df.index)
    col_perm = rng.permutation(as_df.columns)
    df_rand = as_df.loc[row_perm, col_perm]
    return df_rand

test1_df_square = convert_to_df(test1_array)
test2_df_square = convert_to_df(test2_array)

def melt_df(df):
    df_melted = df.reset_index().melt(id_vars="index",
                                var_name="column",
                                value_name="value")
    return df_melted[df_melted["value"] != 0]

test1_df_melted = melt_df(test1_df_square)
test2_df_melted = melt_df(test2_df_square)

test1_clusters_str = [
    {"a", "f", "g", "j"},
    {"b", "c", "e"},
    {"d", "h", "i", "k", "l"}
]

test2_clusters_str = [
    {"d", "e", "f", "g"},
    {"a", "b", "c"}
]

@pytest.mark.parametrize(
    ("input_data", "expected"),
    [
        (test1_df_square, test1_clusters_str),
        (test1_df_melted, test1_clusters_str),
        (test2_df_square, test2_clusters_str),
        (test2_df_melted, test2_clusters_str),
    ]
)

def test_mcl_dataframe_input(input_data, expected):
    result = mcl.markov_cluster(input_data)
    assert result == sorted(expected, key=len, reverse=True)

# Test keep_overlap

test2_overlap_clusters = [
    {3, 4, 5, 6},
    {0, 1, 2, 3}
]

test2_overlap_clusters_str = [
    {"d", "e", "f", "g"},
    {"a", "b", "c", "d"}
]

@pytest.mark.parametrize(
    ("input_data", "expected"),
    [
        (test1_array, test1_clusters),
        (test1_df_square, test1_clusters_str),
        (test1_df_melted, test1_clusters_str),
        (test2_array, test2_overlap_clusters),
        (test2_df_square, test2_overlap_clusters_str),
        (test2_df_melted, test2_overlap_clusters_str),
    ]
)

def test_resolve_overlapping_clusters(input_data, expected):
    result = mcl.markov_cluster(input_data, overlaps="keep")
    assert result == sorted(expected, key=len, reverse=True)

cluster1 = [{0, 1, 2}, {3, 4}]
expected_matrix1 = np.array([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1]
    ])

cluster2 = [{"cat", "dog", "bird"}, {"apple", "banana"}]
expected_matrix2 = pd.DataFrame({
        "apple": [1, 1, 0, 0, 0],
        "banana": [1, 1, 0, 0, 0],
        "bird": [0, 0, 1, 1, 1],
        "cat": [0, 0, 1, 1, 1],
        "dog": [0, 0, 1, 1, 1]
    })

cluster3 = [{0}, {1}, {2}]
expected_matrix3 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

cluster4 = [{0, 1}, {1, 2}]
expected_matrix4 = np.array([
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1]
    ])

cluster5 = []
expected_matrix5 = np.array([])

@pytest.mark.parametrize(
    ("clusters", "expected_matrix"),
    [
        (cluster1, expected_matrix1),
        (cluster2, expected_matrix2),
        (cluster3, expected_matrix3),
        (cluster4, expected_matrix4),
        (cluster5, expected_matrix5)
    ]
)

def test_clusters_to_adjacency(clusters, expected_matrix):
    result = mcl.clusters_to_adjacency(clusters)
    np.testing.assert_array_equal(result, expected_matrix)
