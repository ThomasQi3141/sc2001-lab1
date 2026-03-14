import sys
import time
import random

# 1. Dijkstra's Algorithm Implementation

def dijkstra_matrix_array(graph, src):
    V = len(graph)
    
    # The 'dist' array acts as our Priority Queue data storage
    dist = [sys.maxsize] * V
    dist[src] = 0
    
    # Array to keep track of vertices whose minimum distance is finalized
    visited = [False] * V

    for _ in range(V):
        # --- EXTRACT MINIMUM (Array Priority Queue) ---
        # Linearly scan the array to find the unvisited vertex with the min distance
        min_dist = sys.maxsize
        u = -1
        
        for i in range(V):
            if not visited[i] and dist[i] < min_dist:
                min_dist = dist[i]
                u = i
                
        # If all remaining vertices are unreachable from the source, stop
        if u == -1:
            break
            
        # Mark the chosen vertex as finalized
        visited[u] = True

        # --- RELAXATION (Adjacency Matrix) ---
        # Update distance value of the adjacent vertices of the picked vertex.
        for v in range(V):
            # Update dist[v] only if:
            # 1. v is not visited
            # 2. There is an edge from u to v (graph[u][v] > 0)
            # 3. Total weight of path from src to v through u is smaller than current dist[v]
            if not visited[v] and graph[u][v] > 0:
                if dist[u] + graph[u][v] < dist[v]:
                    dist[v] = dist[u] + graph[u][v]

    return dist

# ==========================================
# 2. Helper: Generate Random Graph
# ==========================================
def generate_random_graph(V, edge_probability=0.5):
    """Generates a random adjacency matrix."""
    graph = [[0 for _ in range(V)] for _ in range(V)]
    for i in range(V):
        for j in range(V):
            if i != j and random.random() < edge_probability:
                graph[i][j] = random.randint(1, 100) # Random weight 1 to 100
    return graph

# ==========================================
# 3. Empirical Analysis / Timing Script
# ==========================================
def empirical_analysis():
    print(f"{'Vertices (|V|)':<15} | {'Edges (~|E|)':<15} | {'Time (Seconds)':<15}")
    print("-" * 50)
    
    # Test different sizes of V to observe the time complexity growth
    vertex_sizes = [100, 250, 500, 750, 1000, 1500, 2000]
    
    for V in vertex_sizes:
        # Keep density at 50% so |E| is roughly 0.5 * V * (V-1)
        graph = generate_random_graph(V, 0.5) 
        
        # Calculate approximate number of edges for printing
        E = sum(1 for i in range(V) for j in range(V) if graph[i][j] > 0)
        
        # Measure execution time
        start_time = time.perf_counter()
        dijkstra_matrix_array(graph, 0)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        print(f"{V:<15} | {E:<15} | {execution_time:.6f}")

if __name__ == "__main__":
    empirical_analysis()