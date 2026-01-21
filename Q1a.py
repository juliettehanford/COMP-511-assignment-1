import os
import zipfile
import requests
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import connected_components

# -------------------------------------------------
# Data acquisition
# -------------------------------------------------

ZIP_URL = "https://networksciencebook.com/translations/en/resources/networks.zip"
DATA_DIR = "networks"

def download_and_extract_datasets():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    zip_path = os.path.join(DATA_DIR, "networks.zip")

    if not os.path.exists(zip_path):
        r = requests.get(ZIP_URL)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(r.content)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)

# -------------------------------------------------
# Graph construction
# -------------------------------------------------

def load_edgelist_as_simple_undirected_graph(path):
    """
    Loads an edge list and returns a simple, undirected,
    unweighted sparse adjacency matrix (CSC).
    """
    edges = np.loadtxt(path, dtype=int)

    # Remove self-loops
    edges = edges[edges[:, 0] != edges[:, 1]]

    # Reindex nodes to 0...N-1
    nodes = np.unique(edges)
    node_map = {node: i for i, node in enumerate(nodes)}
    edges = np.array([[node_map[u], node_map[v]] for u, v in edges])

    n = len(nodes)

    # Create undirected edges (symmetrize)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(rows), dtype=int)

    A = csc_matrix((data, (rows, cols)), shape=(n, n))

    # Remove multi-edges (force binary)
    A.data[:] = 1
    A.eliminate_zeros()

    return A

# -------------------------------------------------
# Reusable computations (Q1a)
# -------------------------------------------------

def graph_size_stats(A):
    """
    Returns:
    - number of nodes
    - number of edges
    """
    n_nodes = A.shape[0]
    n_edges = A.nnz // 2  # undirected
    return n_nodes, n_edges

def connected_component_stats(A):
    """
    Returns:
    - number of connected components
    - size of the largest connected component
    """
    n_components, labels = connected_components(A, directed=False)
    component_sizes = np.bincount(labels)
    giant_size = component_sizes.max()
    return n_components, giant_size

# -------------------------------------------------
# Main analysis
# -------------------------------------------------

def analyze_datasets(dataset_files):
    results = []

    for name, path in dataset_files.items():
        A = load_edgelist_as_simple_undirected_graph(path)

        n_nodes, n_edges = graph_size_stats(A)
        n_components, giant_size = connected_component_stats(A)

        results.append({
            "Dataset": name,
            "Nodes": n_nodes,
            "Edges": n_edges,
            "Connected Components": n_components,
            "Giant Component Size": giant_size
        })

    return pd.DataFrame(results)

# -------------------------------------------------
# Execution
# -------------------------------------------------

if __name__ == "__main__":
    download_and_extract_datasets()

    datasets = {
        "Karate Club": os.path.join(DATA_DIR, "karate.edgelist"),
        "Dolphins": os.path.join(DATA_DIR, "dolphins.edgelist"),
        "Email-Enron": os.path.join(DATA_DIR, "email-Enron.edgelist")
    }

    table = analyze_datasets(datasets)
    print(table.to_string(index=False))