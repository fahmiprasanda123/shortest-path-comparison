import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import (
    dijkstra, 
    bellman_ford, 
    floyd_warshall, 
    johnson,
    NegativeCycleError
)

def create_graph(num_nodes, density=0.5, negative_weights=False):
 
    graph = np.random.rand(num_nodes, num_nodes)
    
    graph[graph > density] = 0
    
    if negative_weights:

        graph[graph > 0] = np.random.randint(-10, 50, graph[graph > 0].shape)
    else:

        graph[graph > 0] = np.random.randint(1, 50, graph[graph > 0].shape)

    np.fill_diagonal(graph, 0)
    

    return csr_matrix(graph)

def run_dijkstra_all_pairs(graph):
    """Menjalankan Dijkstra V kali."""
    num_nodes = graph.shape[0]
    try:
        start_time = time.perf_counter()
        for i in range(num_nodes):

            dijkstra(csgraph=graph, directed=True, indices=i)
        end_time = time.perf_counter()
        return end_time - start_time
    except Exception as e:
        return f"Error: {e}"

def run_bellman_ford_all_pairs(graph):
    """Menjalankan Bellman-Ford V kali."""
    num_nodes = graph.shape[0]
    try:
        start_time = time.perf_counter()
        for i in range(num_nodes):
            bellman_ford(csgraph=graph, directed=True, indices=i)
        end_time = time.perf_counter()
        return end_time - start_time
    except NegativeCycleError:
        return "Negative Cycle Detected"
    except Exception as e:
        return f"Error: {e}"

def run_floyd_warshall(graph):
    """Menjalankan Floyd-Warshall sekali."""
    try:
        start_time = time.perf_counter()
        floyd_warshall(csgraph=graph, directed=True)
        end_time = time.perf_counter()
        return end_time - start_time
    except NegativeCycleError:
        return "Negative Cycle Detected"
    except Exception as e:
        return f"Error: {e}"

def run_johnson(graph):
    """Menjalankan Johnson sekali."""
    try:
        start_time = time.perf_counter()
        johnson(csgraph=graph, directed=True)
        end_time = time.perf_counter()
        return end_time - start_time
    except NegativeCycleError:
        return "Negative Cycle Detected"
    except Exception as e:
        return f"Error: {e}"

# --- DATA UJI DAN EKSEKUSI ---

print("Memulai Uji Perbandingan Algoritma Shortest Path...")
print("-" * 50)


NUM_NODES_DENSE = 150
DENSITY_DENSE = 0.8 

print(f"\nTest 1: Graf Padat (Dense)")
print(f"Nodes: {NUM_NODES_DENSE}, Density: {DENSITY_DENSE}, Bobot: Positif")


graph_dense_pos = create_graph(NUM_NODES_DENSE, DENSITY_DENSE, negative_weights=False)


graph_dense_pos_array = graph_dense_pos.toarray()

time_fw = run_floyd_warshall(graph_dense_pos_array)
time_jh = run_johnson(graph_dense_pos)
time_dk = run_dijkstra_all_pairs(graph_dense_pos)
time_bf = run_bellman_ford_all_pairs(graph_dense_pos)

# APSP: All-Pairs Shortest Path
# V x SSSP: V kali Single-Source Shortest Path

print(f"  Floyd-Warshall (APSP): {time_fw:.4f} detik")
print(f"  Johnson (APSP):        {time_jh:.4f} detik")
print(f"  Dijkstra (V x SSSP):   {time_dk:.4f} detik")
print(f"  Bellman-Ford (V x SSSP):   {time_bf:.4f} detik")
# skip bellman-ford for karena tidak adil karena menggunakan density positif

print("-" * 50)

NUM_NODES_SPARSE = 500
DENSITY_SPARSE = 0.1

print(f"\nTest 2: Graf Jarang (Sparse)")
print(f"Nodes: {NUM_NODES_SPARSE}, Density: {DENSITY_SPARSE}, Bobot: Positif")

# Data Uji 2
graph_sparse_pos = create_graph(NUM_NODES_SPARSE, DENSITY_SPARSE, negative_weights=False)
graph_sparse_pos_array = graph_sparse_pos.toarray()

time_fw = run_floyd_warshall(graph_sparse_pos_array)
time_jh = run_johnson(graph_sparse_pos)
time_dk = run_dijkstra_all_pairs(graph_sparse_pos)
time_bf = run_bellman_ford_all_pairs(graph_sparse_pos)

print(f"  Floyd-Warshall (APSP): {time_fw:.4f} detik")
print(f"  Johnson (APSP):        {time_jh:.4f} detik")
print(f"  Dijkstra (V x SSSP):   {time_dk:.4f} detik")
print(f"  Bellman-Ford (V x SSSP):   {time_bf:.4f} detik")

# skip bellman-ford for karena tidak adil karena menggunakan density positif
print("-" * 50)

# 500 graf nodes sparse dengan bobot negatif
# 10% density
# 25000 edge
NUM_NODES_SPARSE = 500
DENSITY_SPARSE = 0.1

print(f"\nTest 3: Graf Jarang (Sparse)")
print(f"Nodes: {NUM_NODES_SPARSE}, Density: {DENSITY_SPARSE}, Bobot: Negatif")

# Data Uji 3 menggunakan bobot negatif
graph_sparse_neg = create_graph(NUM_NODES_SPARSE, DENSITY_SPARSE, negative_weights=True)
graph_sparse_neg_array = graph_sparse_neg.toarray()

time_fw = run_floyd_warshall(graph_sparse_neg_array)
time_jh = run_johnson(graph_sparse_neg)
time_bf = run_bellman_ford_all_pairs(graph_sparse_neg)

print("  Dijkstra (V x SSSP):   INVALID (Bobot negatif)")



if isinstance(time_fw, float):
    print(f"  Floyd-Warshall (APSP): {time_fw:.4f} detik")
else:
    print(f"  Floyd-Warshall (APSP): {time_fw}")

if isinstance(time_jh, float):
    print(f"  Johnson (APSP):        {time_jh:.4f} detik")
else:
    print(f"  Johnson (APSP):        {time_jh}") 

if isinstance(time_bf, float):
    print(f"  Bellman-Ford (V x SSSP): {time_bf:.4f} detik")
else:
    print(f"  Bellman-Ford (V x SSSP): {time_bf}") 

print("-" * 50)