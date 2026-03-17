import sys
import time
import random
import heapq
import matplotlib.pyplot as plt
import numpy as np


def dijkstra_list_heap(graph, src):
    V = len(graph)
    
    # The 'dist' array acts as our distance tracker
    dist = [sys.maxsize] * V
    dist[src] = 0
    
    # Array to keep track of vertices whose minimum distance is finalized
    visited = [False] * V

    # Priority Queue (Min-Heap) storing tuples of (distance, vertex)
    pq = [(0, src)]

    while pq:
       
        min_dist, u = heapq.heappop(pq)
        
        # If the vertex is already finalized, ignore it (lazy deletion)
        if visited[u]:
            continue
            
        # Mark the chosen vertex as finalized
        visited[u] = True

        
        # Update distance value of the adjacent vertices of the picked vertex.
        for v, weight in graph[u]:
            # Update dist[v] only if:
            # 1. v is not visited
            # 2. Total weight of path from src to v through u is smaller than current dist[v]
            if not visited[v] and dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                
                # Push the newly found shorter distance into the priority queue
                heapq.heappush(pq, (dist[v], v))

    return dist


def generate_random_graph_list(V, edge_probability=0.5):
    """Generates a random adjacency list."""
    graph = [[] for _ in range(V)]
    for i in range(V):
        for j in range(V):
            if i != j and random.random() < edge_probability:
                weight = random.randint(1, 100) # Random weight 1 to 100
                graph[i].append((j, weight))
    return graph


def empirical_analysis():
    print(f"{'Vertices (|V|)':<15} | {'Edges (~|E|)':<15} | {'Time (Seconds)':<15}")
    print("-" * 50)
    
    # Test different sizes of V to observe the time complexity growth
    vertex_sizes = [100, 250, 500, 750, 1000, 1500, 2000]
    
    for V in vertex_sizes:
        # Keep density at 50% so |E| is roughly 0.5 * V * (V-1)
        graph = generate_random_graph_list(V, 0.5) 
        
       
        E = sum(len(adj) for adj in graph)
        
        # Measure execution time
        start_time = time.perf_counter()
        dijkstra_list_heap(graph, 0)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        print(f"{V:<15} | {E:<15} | {execution_time:.6f}")


def plot_runtime_vs_vertices():
    print("\nGenerating Plot 1: Runtime vs Vertices...")
    
    # We test smaller steps to get a smooth curve
    V_values = [50, 100, 200, 300, 400, 500, 600]
    empirical_times = []
    
    for V in V_values:
        runtimes = []
        # Run 5 trials per size to get a stable average
        for _ in range(5):
            graph = generate_random_graph_list(V, 0.5) 
            
            start_time = time.perf_counter()
            dijkstra_list_heap(graph, 0)
            end_time = time.perf_counter()
            
            runtimes.append(end_time - start_time)
            
        empirical_times.append(np.mean(runtimes))

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(V_values, empirical_times, 'g-o', linewidth=2, label='Empirical Runtime (List + Min-Heap)')
    
    plt.xlabel('Number of Vertices (|V|)', fontsize=12)
    plt.ylabel('Average Time (Seconds)', fontsize=12)
    plt.title('Dijkstra Part (b): Runtime vs Vertices (Density = 50%)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def plot_runtime_vs_edges():
    print("\nGenerating Plot 2: Runtime vs Edges...")
    
    V_fixed = 300
    # Vary the density probability from 10% (sparse) to 100% (fully dense)
    probabilities = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    E_values = []
    empirical_times = []
    
    for p in probabilities:
        runtimes = []
        current_E = 0
        
        # Run 5 trials per density to get a stable average
        for _ in range(5):
            graph = generate_random_graph_list(V_fixed, p)
            current_E = sum(len(adj) for adj in graph) # Calculate exact edges
            
            start_time = time.perf_counter()
            dijkstra_list_heap(graph, 0)
            end_time = time.perf_counter()
            
            runtimes.append(end_time - start_time)
            
        E_values.append(current_E)
        empirical_times.append(np.mean(runtimes))

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(E_values, empirical_times, 'b-o', linewidth=2, label=f'Empirical Runtime (Fixed |V|={V_fixed})')
    
    plt.xlabel('Number of Edges (|E|)', fontsize=12)
    plt.ylabel('Average Time (Seconds)', fontsize=12)
    plt.title('Dijkstra Part (b): Runtime vs Edges (Fixed Vertices)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    # 1. Print the raw data table
    empirical_analysis()
    
    # 2. Generate and display the graphs
    plot_runtime_vs_vertices()
    plot_runtime_vs_edges()
