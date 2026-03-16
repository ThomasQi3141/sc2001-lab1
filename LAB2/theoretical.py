import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Complexity models
# -----------------------------
def t_matrix_array(V):
    # Dijkstra with adjacency matrix + array priority queue
    return V**2

def t_adj_heap(V, E, log_base=2):
    # Dijkstra with adjacency list + min-heap
    if log_base == 2:
        return (V + E) * np.log2(V)
    elif log_base == np.e:
        return (V + E) * np.log(V)
    else:
        return (V + E) * (np.log(V) / np.log(log_base))

def crossover_E(V, log_base=2):
    # Solve (V + E) log V = V^2  => E = V^2 / log V - V
    if log_base == 2:
        lg = np.log2(V)
    elif log_base == np.e:
        lg = np.log(V)
    else:
        lg = np.log(V) / np.log(log_base)
    return V**2 / lg - V


# -----------------------------
# Plot 1:
# For several fixed V values, compare T_heap against E
# and show the crossover point
# -----------------------------
V_values = [50, 100, 200, 500]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, V in zip(axes, V_values):
    max_E = V * (V - 1) // 2
    E_vals = np.linspace(0, max_E, 500)

    y_matrix = np.full_like(E_vals, t_matrix_array(V), dtype=float)
    y_heap = t_adj_heap(V, E_vals)

    cross = crossover_E(V)

    ax.plot(E_vals, y_matrix, label=r"$|V|^2$", linewidth=2)
    ax.plot(E_vals, y_heap, label=r"$(|V|+|E|)\log_2 |V|$", linewidth=2)

    if 0 <= cross <= max_E:
        ax.axvline(cross, linestyle="--", linewidth=1.5, label=f"crossover E ≈ {cross:.1f}")
        ax.scatter([cross], [t_matrix_array(V)], s=40)

    ax.set_title(f"V = {V}")
    ax.set_xlabel("|E|")
    ax.set_ylabel("Relative cost")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle("Comparing $|V|^2$ vs $(|V|+|E|)\\log |V|$ for fixed |V|", fontsize=14)
plt.tight_layout()
plt.show()


# -----------------------------
# Plot 2:
# Crossover edge count as V grows
# This tells you when heap becomes slower than matrix
# -----------------------------
V_range = np.arange(2, 1001)
E_cross_vals = crossover_E(V_range)

# For a simple graph, E cannot exceed V(V-1)/2
E_max_vals = V_range * (V_range - 1) / 2

plt.figure(figsize=(10, 6))
plt.plot(V_range, E_cross_vals, label=r"$E_{\mathrm{cross}} = \frac{V^2}{\log_2 V} - V$", linewidth=2)
plt.plot(V_range, E_max_vals, label=r"max simple-graph edges = $\frac{V(V-1)}{2}$", linewidth=2)

# Shade region where crossover is actually achievable in a simple graph
mask = E_cross_vals <= E_max_vals
plt.fill_between(V_range, E_cross_vals, E_max_vals, where=mask, alpha=0.2)

plt.xlabel("|V|")
plt.ylabel("Number of edges")
plt.title("Crossover edge count vs maximum possible edges")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


# -----------------------------
# Plot 3:
# Ratio plot for different graph densities
# ratio = heap / matrix
# ratio < 1 => heap is better
# ratio > 1 => matrix is better
# -----------------------------
V_range = np.arange(2, 1001)

densities = [0.1, 0.25, 0.5, 0.75, 1.0]  # fraction of max edges
plt.figure(figsize=(10, 6))

for p in densities:
    E_vals = p * (V_range * (V_range - 1) / 2)
    ratio = t_adj_heap(V_range, E_vals) / t_matrix_array(V_range)
    plt.plot(V_range, ratio, label=f"density = {p:.2f}")

plt.axhline(1.0, linestyle="--", linewidth=1.5, label="equal cost")
plt.xlabel("|V|")
plt.ylabel(r"$\frac{(|V|+|E|)\log_2 |V|}{|V|^2}$")
plt.title("When is heap-based Dijkstra better or worse?")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()