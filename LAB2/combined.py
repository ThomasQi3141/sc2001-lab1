import sys
import time
import random
import heapq
import math
import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# 1. DIJKSTRA IMPLEMENTATIONS
# =========================================================

def dijkstra_matrix_array(graph_matrix, src):
    """
    Dijkstra using:
    - graph representation: adjacency matrix
    - priority queue: array (linear scan for extract-min)

    Time complexity:
    - extract-min done V times, each O(V)
    - relaxation scans entire row of matrix each time, O(V)
    - total: O(V^2)
    """
    V = len(graph_matrix)

    dist = [sys.maxsize] * V
    dist[src] = 0
    visited = [False] * V

    for _ in range(V):
        # Extract minimum from array in O(V)
        min_dist = sys.maxsize
        u = -1
        for i in range(V):
            if not visited[i] and dist[i] < min_dist:
                min_dist = dist[i]
                u = i

        if u == -1:
            break

        visited[u] = True

        # Relax all possible neighbors by scanning full row in matrix
        for v in range(V):
            weight = graph_matrix[u][v]
            if not visited[v] and weight > 0:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight

    return dist


def dijkstra_list_heap(graph_list, src):
    """
    Dijkstra using:
    - graph representation: adjacency list
    - priority queue: min-heap

    Time complexity:
    - heap operations: O(log V)
    - edges relaxed through adjacency lists
    - total: O((V + E) log V), often written O(E log V)
    """
    V = len(graph_list)

    dist = [sys.maxsize] * V
    dist[src] = 0
    visited = [False] * V

    pq = [(0, src)]

    while pq:
        cur_dist, u = heapq.heappop(pq)

        if visited[u]:
            continue

        visited[u] = True

        for v, weight in graph_list[u]:
            if not visited[v] and dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))

    return dist


# =========================================================
# 2. GRAPH GENERATION / CONVERSION
# =========================================================

def generate_random_graph_both(V, edge_probability=0.5, directed=False, seed=None):
    """
    Generates ONE random weighted graph and returns BOTH:
    - adjacency matrix
    - adjacency list

    This ensures fair comparison because both implementations
    run on the exact same graph.

    We use weight 0 to mean "no edge".

    By default this generates an UNDIRECTED graph, so the maximum
    number of edges is V(V-1)/2.
    """
    if seed is not None:
        random.seed(seed)

    matrix = [[0] * V for _ in range(V)]
    adj_list = [[] for _ in range(V)]

    if directed:
        for i in range(V):
            for j in range(V):
                if i != j and random.random() < edge_probability:
                    weight = random.randint(1, 100)
                    matrix[i][j] = weight
                    adj_list[i].append((j, weight))
    else:
        for i in range(V):
            for j in range(i + 1, V):
                if random.random() < edge_probability:
                    weight = random.randint(1, 100)
                    matrix[i][j] = weight
                    matrix[j][i] = weight
                    adj_list[i].append((j, weight))
                    adj_list[j].append((i, weight))

    return matrix, adj_list


def generate_graph_with_exact_edges_both(V, E, directed=False, seed=None):
    """
    Generate ONE random weighted graph with exactly E edges,
    and return both:
    - adjacency matrix
    - adjacency list

    Default: undirected graph
    Max edges = V(V-1)/2 for undirected
    """
    if seed is not None:
        random.seed(seed)

    if directed:
        max_edges = V * (V - 1)
        if E > max_edges:
            raise ValueError(f"For directed graphs, max edges is {max_edges}")
        all_edges = [(i, j) for i in range(V) for j in range(V) if i != j]
    else:
        max_edges = V * (V - 1) // 2
        if E > max_edges:
            raise ValueError(f"For undirected graphs, max edges is {max_edges}")
        all_edges = [(i, j) for i in range(V) for j in range(i + 1, V)]

    chosen_edges = random.sample(all_edges, E)

    matrix = [[0] * V for _ in range(V)]
    adj_list = [[] for _ in range(V)]

    for u, v in chosen_edges:
        w = random.randint(1, 100)
        matrix[u][v] = w
        adj_list[u].append((v, w))

        if not directed:
            matrix[v][u] = w
            adj_list[v].append((u, w))

    return matrix, adj_list


def count_edges_matrix(matrix, directed=False):
    V = len(matrix)
    edges = 0

    for i in range(V):
        for j in range(V):
            if matrix[i][j] > 0:
                edges += 1

    if not directed:
        edges //= 2

    return edges


# =========================================================
# 3. TIMING HELPERS
# =========================================================

def time_function(func, *args):
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return result, end - start


def average_runtime(func, graph, src=0, trials=3):
    runtimes = []
    for _ in range(trials):
        _, t = time_function(func, graph, src)
        runtimes.append(t)
    return np.mean(runtimes)


# =========================================================
# 4. PART (a): MATRIX + ARRAY EXPERIMENTS
# =========================================================

def empirical_analysis_part_a():
    print("\nPART (a): Adjacency Matrix + Array")
    print(f"{'Vertices (|V|)':<15} | {'Edges (|E|)':<15} | {'Avg Time (s)':<15}")
    print("-" * 55)

    vertex_sizes = [100, 250, 500, 750, 1000]
    density = 0.5

    for V in vertex_sizes:
        matrix, _ = generate_random_graph_both(V, density, directed=False)
        E = count_edges_matrix(matrix, directed=False)
        avg_time = average_runtime(dijkstra_matrix_array, matrix, 0, trials=3)
        print(f"{V:<15} | {E:<15} | {avg_time:.6f}")


def plot_part_a_runtime_vs_vertices():
    print("\nGenerating Part (a) Plot: Runtime vs |V|")

    V_values = [50, 100, 150, 200, 300, 400, 500, 600]
    density = 0.5
    times = []

    for V in V_values:
        matrix, _ = generate_random_graph_both(V, density, directed=False)
        avg_time = average_runtime(dijkstra_matrix_array, matrix, 0, trials=5)
        times.append(avg_time)

    plt.figure(figsize=(10, 6))
    plt.plot(V_values, times, marker='o', linewidth=2, label='Matrix + Array')
    plt.xlabel('Number of Vertices (|V|)')
    plt.ylabel('Average Runtime (seconds)')
    plt.title('Part (a): Dijkstra Runtime vs |V| (Adjacency Matrix + Array)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_part_a_runtime_vs_edges():
    print("\nGenerating Part (a) Plot: Runtime vs |E|")

    V_fixed = 300
    probabilities = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    E_values = []
    times = []

    for p in probabilities:
        trial_times = []
        edge_counts = []

        for _ in range(5):
            matrix, _ = generate_random_graph_both(V_fixed, p, directed=False)
            E = count_edges_matrix(matrix, directed=False)
            _, t = time_function(dijkstra_matrix_array, matrix, 0)

            edge_counts.append(E)
            trial_times.append(t)

        E_values.append(np.mean(edge_counts))
        times.append(np.mean(trial_times))

    plt.figure(figsize=(10, 6))
    plt.plot(E_values, times, marker='o', linewidth=2, label=f'Matrix + Array (|V|={V_fixed})')
    plt.xlabel('Number of Edges (|E|)')
    plt.ylabel('Average Runtime (seconds)')
    plt.title('Part (a): Dijkstra Runtime vs |E| (Fixed |V|)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 5. PART (b): LIST + HEAP EXPERIMENTS
# =========================================================

def empirical_analysis_part_b():
    print("\nPART (b): Adjacency List + Min-Heap")
    print(f"{'Vertices (|V|)':<15} | {'Edges (|E|)':<15} | {'Avg Time (s)':<15}")
    print("-" * 55)

    vertex_sizes = [100, 250, 500, 750, 1000, 1500, 2000]
    density = 0.5

    for V in vertex_sizes:
        matrix, adj_list = generate_random_graph_both(V, density, directed=False)
        E = count_edges_matrix(matrix, directed=False)
        avg_time = average_runtime(dijkstra_list_heap, adj_list, 0, trials=3)
        print(f"{V:<15} | {E:<15} | {avg_time:.6f}")


def plot_part_b_runtime_vs_vertices():
    print("\nGenerating Part (b) Plot: Runtime vs |V|")

    V_values = [50, 100, 200, 300, 400, 500, 600]
    density = 0.5
    times = []

    for V in V_values:
        _, adj_list = generate_random_graph_both(V, density, directed=False)
        avg_time = average_runtime(dijkstra_list_heap, adj_list, 0, trials=5)
        times.append(avg_time)

    plt.figure(figsize=(10, 6))
    plt.plot(V_values, times, marker='o', linewidth=2, label='Adjacency List + Min-Heap')
    plt.xlabel('Number of Vertices (|V|)')
    plt.ylabel('Average Runtime (seconds)')
    plt.title('Part (b): Dijkstra Runtime vs |V| (Adjacency List + Min-Heap)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_part_b_runtime_vs_edges():
    print("\nGenerating Part (b) Plot: Runtime vs |E|")

    V_fixed = 300
    probabilities = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    E_values = []
    times = []

    for p in probabilities:
        trial_times = []
        edge_counts = []

        for _ in range(5):
            matrix, adj_list = generate_random_graph_both(V_fixed, p, directed=False)
            E = count_edges_matrix(matrix, directed=False)
            _, t = time_function(dijkstra_list_heap, adj_list, 0)

            edge_counts.append(E)
            trial_times.append(t)

        E_values.append(np.mean(edge_counts))
        times.append(np.mean(trial_times))

    plt.figure(figsize=(10, 6))
    plt.plot(E_values, times, marker='o', linewidth=2, label=f'Adjacency List + Min-Heap (|V|={V_fixed})')
    plt.xlabel('Number of Edges (|E|)')
    plt.ylabel('Average Runtime (seconds)')
    plt.title('Part (b): Dijkstra Runtime vs |E| (Fixed |V|)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 6. PART (c): DIRECT COMPARISON
# =========================================================

def empirical_comparison_part_c():
    print("\nPART (c): Direct Comparison (Fixed |V| = 1000)")
    print(
        f"{'Density':<10} | {'|E|':<12} | {'Matrix+Array (s)':<18} | "
        f"{'List+Heap (s)':<15} | {'Faster'}"
    )
    print("-" * 85)

    V = 1000
    probabilities = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    for p in probabilities:
        matrix, adj_list = generate_random_graph_both(V, p, directed=False)
        E = count_edges_matrix(matrix, directed=False)

        time_a = average_runtime(dijkstra_matrix_array, matrix, 0, trials=3)
        time_b = average_runtime(dijkstra_list_heap, adj_list, 0, trials=3)

        faster = "Matrix+Array" if time_a < time_b else "List+Heap"
        print(f"{p:<10.2f} | {E:<12} | {time_a:<18.6f} | {time_b:<15.6f} | {faster}")


def plot_comparison_vs_vertices():
    print("\nGenerating Part (c) Comparison Plot: Runtime vs |V|")

    V_values = [50, 100, 150, 200, 300, 400, 500, 600]
    density = 0.5

    times_a = []
    times_b = []

    for V in V_values:
        matrix, adj_list = generate_random_graph_both(V, density, directed=False)
        times_a.append(average_runtime(dijkstra_matrix_array, matrix, 0, trials=5))
        times_b.append(average_runtime(dijkstra_list_heap, adj_list, 0, trials=5))

    plt.figure(figsize=(10, 6))
    plt.plot(V_values, times_a, marker='o', linewidth=2, label='Matrix + Array')
    plt.plot(V_values, times_b, marker='s', linewidth=2, label='List + Min-Heap')
    plt.xlabel('Number of Vertices (|V|)')
    plt.ylabel('Average Runtime (seconds)')
    plt.title('Part (c): Comparison of Two Dijkstra Implementations vs |V|')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_comparison_vs_edges():
    print("\nGenerating Part (c) Comparison Plot: Runtime vs |E| (Fixed |V| = 1000)")

    V_fixed = 1000
    probabilities = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    E_values = []
    times_a = []
    times_b = []

    for p in probabilities:
        edge_counts = []
        runtimes_a = []
        runtimes_b = []

        for _ in range(3):
            matrix, adj_list = generate_random_graph_both(V_fixed, p, directed=False)
            E = count_edges_matrix(matrix, directed=False)

            _, t_a = time_function(dijkstra_matrix_array, matrix, 0)
            _, t_b = time_function(dijkstra_list_heap, adj_list, 0)

            edge_counts.append(E)
            runtimes_a.append(t_a)
            runtimes_b.append(t_b)

        E_values.append(np.mean(edge_counts))
        times_a.append(np.mean(runtimes_a))
        times_b.append(np.mean(runtimes_b))

    plt.figure(figsize=(10, 6))
    plt.plot(E_values, times_a, marker='o', linewidth=2, label='Matrix + Array')
    plt.plot(E_values, times_b, marker='s', linewidth=2, label='List + Min-Heap')
    plt.xlabel('Number of Edges (|E|)')
    plt.ylabel('Average Runtime (seconds)')
    plt.title(f'Part (c): Comparison of Two Dijkstra Implementations vs |E| (Fixed |V|={V_fixed})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_comparison_by_density():
    """
    Extra graph for part (c):
    compares both implementations as graph density increases,
    while keeping |V| fixed at 1000.
    """
    print("\nGenerating Part (c) Comparison Plot: Runtime vs Density (Fixed |V| = 1000)")

    V_fixed = 1000
    probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    times_a = []
    times_b = []

    for p in probabilities:
        runtimes_a = []
        runtimes_b = []

        for _ in range(3):
            matrix, adj_list = generate_random_graph_both(V_fixed, p, directed=False)
            _, t_a = time_function(dijkstra_matrix_array, matrix, 0)
            _, t_b = time_function(dijkstra_list_heap, adj_list, 0)

            runtimes_a.append(t_a)
            runtimes_b.append(t_b)

        times_a.append(np.mean(runtimes_a))
        times_b.append(np.mean(runtimes_b))

    plt.figure(figsize=(10, 6))
    plt.plot(probabilities, times_a, marker='o', linewidth=2, label='Matrix + Array')
    plt.plot(probabilities, times_b, marker='s', linewidth=2, label='List + Min-Heap')
    plt.xlabel('Edge Probability / Graph Density')
    plt.ylabel('Average Runtime (seconds)')
    plt.title(f'Part (c): Comparison vs Graph Density (Fixed |V|={V_fixed})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def empirical_dense_cases_part_c():
    """
    Extra dense test cases for part (c), using undirected graphs.
    These are intended to check whether matrix+array becomes competitive
    when edge density is close to maximum.
    """
    print("\nPART (c): Extra Dense Test Cases (Undirected)")
    print(
        f"{'|V|':<8} | {'Density':<8} | {'|E|':<12} | {'Max |E|':<12} | "
        f"{'Matrix+Array (s)':<18} | {'List+Heap (s)':<15} | {'Faster'}"
    )
    print("-" * 110)

    vertex_sizes = [200, 300, 400, 500]
    densities = [0.90, 0.95, 0.98, 1.00]

    for V in vertex_sizes:
        max_edges = V * (V - 1) // 2

        for p in densities:
            runtimes_a = []
            runtimes_b = []
            edge_counts = []

            for _ in range(5):
                matrix, adj_list = generate_random_graph_both(
                    V,
                    edge_probability=p,
                    directed=False
                )

                E = count_edges_matrix(matrix, directed=False)

                _, t_a = time_function(dijkstra_matrix_array, matrix, 0)
                _, t_b = time_function(dijkstra_list_heap, adj_list, 0)

                edge_counts.append(E)
                runtimes_a.append(t_a)
                runtimes_b.append(t_b)

            avg_E = np.mean(edge_counts)
            avg_a = np.mean(runtimes_a)
            avg_b = np.mean(runtimes_b)
            faster = "Matrix+Array" if avg_a < avg_b else "List+Heap"

            print(f"{V:<8} | {p:<8.2f} | {avg_E:<12.0f} | {max_edges:<12} | {avg_a:<18.6f} | {avg_b:<15.6f} | {faster}")


def plot_comparison_near_max_density():
    """
    Compare both implementations on UNDIRECTED graphs whose density
    is very close to the maximum possible number of edges.

    For undirected graphs, max edges = V(V-1)/2.
    """
    print("\nGenerating Part (c) Dense Comparison Plot: Near-Max Edge Density")

    V_fixed = 400
    probabilities = [0.85, 0.90, 0.95, 0.98, 1.00]

    E_values = []
    times_a = []
    times_b = []

    for p in probabilities:
        edge_counts = []
        runtimes_a = []
        runtimes_b = []

        for _ in range(7):
            matrix, adj_list = generate_random_graph_both(
                V_fixed,
                edge_probability=p,
                directed=False
            )

            E = count_edges_matrix(matrix, directed=False)

            _, t_a = time_function(dijkstra_matrix_array, matrix, 0)
            _, t_b = time_function(dijkstra_list_heap, adj_list, 0)

            edge_counts.append(E)
            runtimes_a.append(t_a)
            runtimes_b.append(t_b)

        E_values.append(np.mean(edge_counts))
        times_a.append(np.mean(runtimes_a))
        times_b.append(np.mean(runtimes_b))

    max_edges = V_fixed * (V_fixed - 1) // 2
    print(f"\nNear-max density experiment (undirected, |V|={V_fixed})")
    print(f"Maximum possible edges = {max_edges}")
    print(
        f"{'Density':<10} | {'Avg |E|':<12} | {'Matrix+Array (s)':<18} | "
        f"{'List+Heap (s)':<15} | {'Faster'}"
    )
    print("-" * 85)

    for p, E, ta, tb in zip(probabilities, E_values, times_a, times_b):
        faster = "Matrix+Array" if ta < tb else "List+Heap"
        print(f"{p:<10.2f} | {E:<12.0f} | {ta:<18.6f} | {tb:<15.6f} | {faster}")

    plt.figure(figsize=(10, 6))
    plt.plot(E_values, times_a, marker='o', linewidth=2, label='Matrix + Array')
    plt.plot(E_values, times_b, marker='s', linewidth=2, label='List + Min-Heap')
    plt.xlabel('Number of Edges (|E|)')
    plt.ylabel('Average Runtime (seconds)')
    plt.title(f'Part (c): Near-Max Density Comparison (Undirected, |V|={V_fixed})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 6B. SLIDE-STYLE REPRODUCTION OF EMPIRICAL RESULT
# =========================================================

def empirical_result_reproduction():
    """
    Recreate the empirical result style from the slide:
    - fixed |V| = 500
    - undirected graphs
    - exact chosen edge counts
    """
    print("\nRecreating empirical result for Part (c) ...")

    V = 5000
    max_edges = V * (V - 1) // 2
    edge_values = [100000, 500000, 2000000, 5000000, 8000000, 10000000, 11500000, 12000000, 12300000, 12400000, 12497500]

    print(f"\nFixed |V| = {V}, maximum undirected edges = {max_edges}")
    print(
        f"{'|E|':<10} | {'Density':<16} | {'Matrix+Array (s)':<18} | "
        f"{'List+Heap (s)':<15} | {'Faster'}"
    )
    print("-" * 85)

    E_values = []
    times_a = []
    times_b = []

    for E in edge_values:
        runtimes_a = []
        runtimes_b = []

        for _ in range(5):
            matrix, adj_list = generate_graph_with_exact_edges_both(
                V=V,
                E=E,
                directed=False
            )

            _, t_a = time_function(dijkstra_matrix_array, matrix, 0)
            _, t_b = time_function(dijkstra_list_heap, adj_list, 0)

            runtimes_a.append(t_a)
            runtimes_b.append(t_b)

        avg_a = np.mean(runtimes_a)
        avg_b = np.mean(runtimes_b)

        density_pct = 100.0 * E / max_edges
        faster = "Matrix+Array" if avg_a < avg_b else "List+Heap"

        print(
            f"{E:<10} | "
            f"{density_pct:>5.1f}% density   | "
            f"{avg_a:<18.6f} | "
            f"{avg_b:<15.6f} | "
            f"{faster}"
        )

        E_values.append(E)
        times_a.append(avg_a)
        times_b.append(avg_b)

    return V, max_edges, E_values, times_a, times_b


def plot_empirical_result_reproduction():
    V, max_edges, E_values, times_a, times_b = empirical_result_reproduction()

    # Find first crossover point where matrix becomes <= heap
    crossover_edge = None
    for E, ta, tb in zip(E_values, times_a, times_b):
        if ta <= tb:
            crossover_edge = E
            break

    plt.figure(figsize=(10, 6))
    plt.plot(
        E_values, times_a,
        'o-r',
        linewidth=2,
        label='Part A: Matrix + Array PQ (O(|V|²))'
    )
    plt.plot(
        E_values, times_b,
        's-b',
        linewidth=2,
        label='Part B: List + Heap PQ (O((|V| + |E|)log|V|))'
    )

    if crossover_edge is not None:
        plt.axvline(
            x=crossover_edge,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            label=f'Crossover ≈ {crossover_edge} edges'
        )

    plt.xlabel(f'Number of Edges |E| (|V| = {V} fixed)')
    plt.ylabel('Average Runtime (s)')
    plt.title(f'Part C: Runtime Comparison of Dijkstra’s Implementations (|V| = {V})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 7. OPTIONAL CORRECTNESS CHECK
# =========================================================

def verify_same_results():
    """
    Optional sanity check:
    generate several random graphs and confirm both implementations
    produce identical shortest path arrays.
    """
    print("\nVerifying correctness between both implementations...")

    for test_id in range(5):
        V = 20
        density = random.choice([0.2, 0.4, 0.6, 0.8])
        matrix, adj_list = generate_random_graph_both(V, density, directed=False)

        dist_a = dijkstra_matrix_array(matrix, 0)
        dist_b = dijkstra_list_heap(adj_list, 0)

        if dist_a != dist_b:
            print(f"Mismatch found on test {test_id + 1}!")
            print("Matrix+Array:", dist_a)
            print("List+Heap:", dist_b)
            return

    print("All verification tests passed. Both implementations produce the same results.")


# =========================================================
# 8. MAIN
# =========================================================

if __name__ == "__main__":
    # # Optional: verify correctness first
    # verify_same_results()

    # # Part (a)
    # empirical_analysis_part_a()
    # plot_part_a_runtime_vs_vertices()
    # plot_part_a_runtime_vs_edges()

    # # Part (b)
    # empirical_analysis_part_b()
    # plot_part_b_runtime_vs_vertices()
    # plot_part_b_runtime_vs_edges()

    # # Part (c)
    # empirical_comparison_part_c()
    # plot_comparison_vs_vertices()
    # plot_comparison_vs_edges()
    # plot_comparison_by_density()

    # # Extra dense cases for part (c)
    # empirical_dense_cases_part_c()
    # plot_comparison_near_max_density()

    # Slide-style reproduction
    plot_empirical_result_reproduction()