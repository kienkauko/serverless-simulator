import time
import networkx as nx
import random

# Configuration
NUM_NODES = 1000
NUM_EDGES = 5000
NUM_ITERATIONS = 100000

print(f"Testing with {NUM_NODES} nodes, {NUM_EDGES} edges, {NUM_ITERATIONS} iterations")
print("-" * 80)

# Setup NetworkX Graph
G = nx.DiGraph()

# Add nodes
for i in range(NUM_NODES):
    G.add_node(str(i))

# Add edges with attributes (similar to your topology)
edges_list = []
for _ in range(NUM_EDGES):
    n1 = str(random.randint(0, NUM_NODES - 1))
    n2 = str(random.randint(0, NUM_NODES - 1))
    if n1 != n2:
        bandwidth = random.uniform(100, 1000)
        latency = random.uniform(0.001, 0.05)
        level = f"{random.randint(1, 3)}-{random.randint(1, 3)}"
        G.add_edge(n1, n2, bandwidth=bandwidth, latency=latency, level=level)
        edges_list.append((n1, n2))

# Setup Dictionary-based storage
edge_dict = {}
for n1, n2, data in G.edges(data=True):
    edge_dict[(n1, n2)] = {
        'bandwidth': data['bandwidth'],
        'latency': data['latency'],
        'level': data['level']
    }

print(f"Actual number of edges: {len(edges_list)}")
print()

# ============================================================================
# Test 1: Reading all three attributes (bandwidth, latency, level)
# ============================================================================
print("TEST 1: Reading all three attributes (bandwidth, latency, level)")
print("-" * 80)

# Method 1: NetworkX with get_edge_data()
start = time.time()
for _ in range(NUM_ITERATIONS):
    edge = random.choice(edges_list)
    data = G.get_edge_data(edge[0], edge[1])
    bw = data['bandwidth']
    lat = data['latency']
    lvl = data['level']
nx_get_edge_data_time = time.time() - start
print(f"NetworkX (get_edge_data):     {nx_get_edge_data_time:.4f}s")

# Method 2: NetworkX with direct access G[n1][n2]
start = time.time()
for _ in range(NUM_ITERATIONS):
    edge = random.choice(edges_list)
    data = G[edge[0]][edge[1]]
    bw = data['bandwidth']
    lat = data['latency']
    lvl = data['level']
nx_direct_time = time.time() - start
print(f"NetworkX (direct G[n1][n2]):  {nx_direct_time:.4f}s")

# Method 3: Separate Dictionary
start = time.time()
for _ in range(NUM_ITERATIONS):
    edge = random.choice(edges_list)
    data = edge_dict[edge]
    bw = data['bandwidth']
    lat = data['latency']
    lvl = data['level']
dict_time = time.time() - start
print(f"Dictionary (edge_dict):        {dict_time:.4f}s")

print()
print(f"Speedup (dict vs get_edge_data): {nx_get_edge_data_time / dict_time:.2f}x faster")
print(f"Speedup (dict vs direct):        {nx_direct_time / dict_time:.2f}x faster")
print()

# ============================================================================
# Test 2: Reading single attribute (bandwidth only)
# ============================================================================
print("TEST 2: Reading single attribute (bandwidth only)")
print("-" * 80)

# Method 1: NetworkX with get_edge_data()
start = time.time()
for _ in range(NUM_ITERATIONS):
    edge = random.choice(edges_list)
    bw = G.get_edge_data(edge[0], edge[1])['bandwidth']
nx_get_single_time = time.time() - start
print(f"NetworkX (get_edge_data):     {nx_get_single_time:.4f}s")

# Method 2: NetworkX with direct access
start = time.time()
for _ in range(NUM_ITERATIONS):
    edge = random.choice(edges_list)
    bw = G[edge[0]][edge[1]]['bandwidth']
nx_direct_single_time = time.time() - start
print(f"NetworkX (direct G[n1][n2]):  {nx_direct_single_time:.4f}s")

# Method 3: Separate Dictionary
start = time.time()
for _ in range(NUM_ITERATIONS):
    edge = random.choice(edges_list)
    bw = edge_dict[edge]['bandwidth']
dict_single_time = time.time() - start
print(f"Dictionary (edge_dict):        {dict_single_time:.4f}s")

print()
print(f"Speedup (dict vs get_edge_data): {nx_get_single_time / dict_single_time:.2f}x faster")
print(f"Speedup (dict vs direct):        {nx_direct_single_time / dict_single_time:.2f}x faster")
print()

# ============================================================================
# Test 3: Updating attributes (simulating flow registration)
# ============================================================================
print("TEST 3: Updating attributes (simulating num_active_flows increment)")
print("-" * 80)

# Add num_active_flows to both storage methods
for n1, n2 in G.edges():
    G[n1][n2]['num_active_flows'] = 0

for edge_key in edge_dict:
    edge_dict[edge_key]['num_active_flows'] = 0

# Method 1: NetworkX update
start = time.time()
for _ in range(NUM_ITERATIONS):
    edge = random.choice(edges_list)
    G[edge[0]][edge[1]]['num_active_flows'] += 1
nx_update_time = time.time() - start
print(f"NetworkX (G[n1][n2]['attr']):  {nx_update_time:.4f}s")

# Method 2: Dictionary update
start = time.time()
for _ in range(NUM_ITERATIONS):
    edge = random.choice(edges_list)
    edge_dict[edge]['num_active_flows'] += 1
dict_update_time = time.time() - start
print(f"Dictionary (dict[edge]['attr']): {dict_update_time:.4f}s")

print()
print(f"Speedup (dict vs NetworkX):      {nx_update_time / dict_update_time:.2f}x faster")
print()

# ============================================================================
# Test 4: Iteration over all edges (checking bandwidth)
# ============================================================================
print("TEST 4: Iteration over all edges")
print("-" * 80)

# Method 1: NetworkX iteration
start = time.time()
for _ in range(100):  # Fewer iterations since this is slower
    total_bw = 0
    for n1, n2, data in G.edges(data=True):
        total_bw += data['bandwidth']
nx_iter_time = time.time() - start
print(f"NetworkX (edges(data=True)):   {nx_iter_time:.4f}s")

# Method 2: Dictionary iteration
start = time.time()
for _ in range(100):
    total_bw = 0
    for edge, data in edge_dict.items():
        total_bw += data['bandwidth']
dict_iter_time = time.time() - start
print(f"Dictionary (items()):           {dict_iter_time:.4f}s")

print()
print(f"Speedup (dict vs NetworkX):      {nx_iter_time / dict_iter_time:.2f}x faster")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"For your simulation hot paths (bandwidth allocation, flow tracking):")
print(f"  - Reading attributes:  Dictionary is ~{nx_direct_time / dict_time:.1f}x faster")
print(f"  - Updating attributes: Dictionary is ~{nx_update_time / dict_update_time:.1f}x faster")
print(f"  - Iterating edges:     Dictionary is ~{nx_iter_time / dict_iter_time:.1f}x faster")
print()
print("RECOMMENDATION:")
print("  Keep static attributes (latency, level) in NetworkX for graph algorithms")
print("  Move dynamic attributes to separate dicts for performance-critical operations")