import networkx as nx
import matplotlib.pyplot as plt
import math # Needed for edge label positioning if using my_draw...

# Dummy function for my_draw_networkx_edge_labels if you don't have it
# Replace with your actual implementation
def my_draw_networkx_edge_labels(G, pos, edge_labels=None, label_pos=0.5,
                                font_size=10, font_color='k', font_family='sans-serif',
                                font_weight='normal', alpha=None, bbox=None,
                                horizontalalignment='center', verticalalignment='center',
                                ax=None, rotate=True, clip_on=True, rad=0):
    """Draw edge labels."""
    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (u, v), label in labels.items():
        if (u, v) not in pos or (v, u) not in pos: # Check if edge exists in pos
             if u not in pos or v not in pos:
                 print(f"Warning: Node {u} or {v} not in pos dictionary for edge label.")
                 continue # Skip if nodes aren't positioned

        (x1, y1) = pos[u]
        (x2, y2) = pos[v]
        (x, y) = (x1 * label_pos + x2 * (1.0 - label_pos),
                  y1 * label_pos + y2 * (1.0 - label_pos))
        pos_1 = ax.transData.transform(np.array(pos[u]))
        pos_2 = ax.transData.transform(np.array(pos[v]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        # Calculate offset for curved edges
        if rad != 0:
             dx = x2 - x1
             dy = y2 - y1
             norm = np.sqrt(dx*dx + dy*dy)
             if norm > 0: # Avoid division by zero
                 # Perpendicular vector
                 pdx = -dy / norm
                 pdy = dx / norm
                 # Approximate arc height
                 arc_height = norm * rad * 0.5 # Adjust multiplier as needed
                 # Offset midpoint
                 offset_x, offset_y = pdx * arc_height, pdy * arc_height
                 mid_x, mid_y = (x1 + x2)/2 + offset_x, (y1+y2)/2 + offset_y
                 (x,y) = (mid_x, mid_y)


        if rotate:
            angle = math.atan2(y2 - y1, x2 - x1) / (2.0 * math.pi) * 360
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            x, y = np.dot(ax.transData.transform(np.array((x, y))),
                          np.linalg.inv(ax.transData.transform(np.array((0, 0)))))
            transform = ax.transData
        else:
            angle = 0.0
            transform = ax.transData

        trans = transform.transform((x, y))

        t = ax.text(x, y, label, size=font_size, color=font_color,
                    family=font_family, weight=font_weight,
                    horizontalalignment=horizontalalignment,
                    verticalalignment=verticalalignment,
                    transform=transform,
                    bbox=bbox, zorder=1, clip_on=clip_on)
        text_items[(u, v)] = t

    ax.autoscale_view()
    return text_items

# --- Updated Drawing Function ---

def draw_graph_updated(G, node_size=1500, rad=-0.2, scale_x=2.0, scale_y=2.0, k_offset_x=0.3, k_offset_y=0.3):
    """
    Draws the state transition diagram described by the NetworkX graph G.
    States are expected to be tuples (i, j, k) where i, j, k >= 0.

    Layout Rules:
    - The base position is determined by (i, j).
    - Increasing 'i' moves nodes horizontally to the right (positive x).
    - Increasing 'j' moves nodes vertically downwards (negative y).
    - Increasing 'k' offsets nodes diagonally (default: up-right) relative
      to their (i, j) base position.

    Resulting Transitions (based on default parameters):
    - (i, j, k) -> (i+1, j, k): Moves horizontally right (x increases by scale_x).
    - (i, j, k) -> (i-1, j+1, k): Moves diagonally down-left (x decreases by scale_x, y decreases by scale_y).
                                   This corresponds to the user's "(i decr, j incr) -> vertical" sequences like (2,0,0)->(1,1,0)->(0,2,0).
    - (i, j, k) -> (i, j-1, k+1): Moves diagonally up-right (x increases by k_offset_x, y increases by (scale_y + k_offset_y)).
                                   This corresponds to the user's "(j decr, k incr) -> diagonal" transition like (0,1,0)->(0,0,1).

    Args:
        G (networkx.DiGraph): State transition graph. Nodes MUST be tuples (i, j, k).
                              Assumes G might have attributes G.labels, G.edge_labels, G.edge_cols.
                              If not present, defaults will be used.
        node_size (int): Size of a node in the diagram.
        rad (float): Curvature of the edges (for arc3 connectionstyle). Use 0 for straight lines.
        scale_x (float): Base horizontal spacing factor between nodes based on 'i'.
        scale_y (float): Base vertical spacing factor between nodes based on 'j'.
        k_offset_x (float): Horizontal offset added per unit increase in 'k'.
        k_offset_y (float): Vertical offset added per unit increase in 'k'.
                            Positive value offsets slightly 'up' relative to the base -j position.
    """

    # --- 1. Calculate Node Positions ---
    pos = {}
    # max_queue = G.max_queue # Assuming this attribute exists if needed elsewhere, not directly used for positioning here.
    for node in G.nodes():
        if isinstance(node, tuple) and len(node) == 3:
            try:
                # Ensure components are numeric and non-negative
                i, j, k = map(float, node) # Convert to float for calculation
                if not (i >= 0 and j >= 0 and k >= 0):
                     raise ValueError("Components must be non-negative")
                i_int, j_int, k_int = int(i), int(j), int(k) # Keep int versions if needed later

            except (ValueError, TypeError):
                print(f"Warning: Node {node} has invalid or non-numeric components. Skipping positioning.")
                pos[node] = (0, 0) # Assign default position or handle as error
                continue

            # Base position: i controls x, j controls y (negated for downward)
            base_x = scale_x * i
            base_y = -scale_y * j # Negative j moves nodes down

            # Apply offset based on k for the 'third dimension' effect
            final_x = base_x + k * k_offset_x
            final_y = base_y + k * k_offset_y

            pos[node] = (final_x, final_y)
        else:
            # Handle nodes not in the expected (i, j, k) tuple format
            print(f"Warning: Node '{node}' is not in (i, j, k) format. Placing at (0,0).")
            pos[node] = (0, 0) # Assign a default position

    # --- 2. Prepare for Drawing ---
    plt.figure(figsize=(14, 10), clear=True) # Increased default figure size slightly
    ax = plt.gca() # Get axes for edge label function

    # --- 3. Draw the Graph Components ---
    # Use G attributes if they exist, otherwise provide defaults
    labels = getattr(G, 'labels', {node: str(node) for node in G.nodes()}) # Default: string representation of node
    edge_labels = getattr(G, 'edge_labels', {(u,v): f"{u}->{v}" for u,v in G.edges()}) # Default: show connection
    edge_cols = getattr(G, 'edge_cols', {edge: 'black' for edge in G.edges()}) # Default: black edges

    # Filter out edges where nodes weren't positioned correctly
    valid_edges = [(u, v) for u, v in G.edges() if u in pos and v in pos]
    filtered_edge_cols = [edge_cols.get(edge, 'red') for edge in valid_edges] # Use red for missing colors

    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_shape ='o', node_size=node_size, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_color='black', ax=ax) # Use provided/default labels

    # Only draw edges that have valid start/end node positions
    if valid_edges:
         edges_drawn = nx.draw_networkx_edges(G, pos, edgelist=valid_edges, width=1, edge_color=filtered_edge_cols,
                               node_size=node_size,
                               arrows=True, arrowstyle='-|>',
                               connectionstyle=f"arc3,rad={rad}", ax=ax)

         # Draw edge labels using the custom function (or nx default)
         # Filter edge labels to only include those for valid edges
         valid_edge_labels = {edge: label for edge, label in edge_labels.items() if edge in valid_edges}
         if valid_edge_labels:
              try:
                   # Using the potentially custom function
                   my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=valid_edge_labels, rotate=False, rad=rad, font_size=8)
              except NameError:
                   # Fallback to networkx default if my_draw... not defined
                   print("Warning: my_draw_networkx_edge_labels not defined, using nx.draw_networkx_edge_labels.")
                   nx.draw_networkx_edge_labels(G, pos, edge_labels=valid_edge_labels, label_pos=0.4,
                                             font_size=8, ax=ax, rotate=False)

    plt.axis('off')
    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show()

# --- Example Usage ---
import numpy as np # Required for my_draw_networkx_edge_labels dummy

# Create a sample graph
G = nx.DiGraph()

# Add nodes following the (i, j, k) structure
nodes_to_add = [
    (0,0,0), (1,0,0), (2,0,0),
    (0,1,0), (1,1,0),
    (0,2,0),
    (0,0,1), (1,0,1),
    (0,1,1)
]
G.add_nodes_from(nodes_to_add)

# Add edges representing the transitions
edges_to_add = [
    ((0,0,0), (1,0,0)), # i increases -> horizontal right
    ((1,0,0), (2,0,0)), # i increases -> horizontal right
    ((1,0,0), (0,1,0)), # i decreases, j increases -> diagonal down-left ("vertical" sequence)
    ((2,0,0), (1,1,0)), # i decreases, j increases -> diagonal down-left
    ((1,1,0), (0,2,0)), # i decreases, j increases -> diagonal down-left
    ((0,1,0), (0,0,1)), # j decreases, k increases -> diagonal up-right ("diagonal")
    ((1,0,0), (1,0,1)), # k increases -> offset diagonally up-right
    ((0,1,0), (0,1,1)), # k increases -> offset diagonally up-right
    ((1,1,0), (1,0,1)), # j decreases, k increases -> diagonal up-right
    ((0,0,1), (1,0,1)), # i increases -> horizontal right
    ((0,1,1), (1,1,0))  # Example of a different transition
]
G.add_edges_from(edges_to_add)

# Add dummy attributes (replace with your actual data)
G.labels = {node: f"{node}" for node in G.nodes()}
G.edge_labels = {edge: f"{edge[0][0]}{edge[0][1]}{edge[0][2]}\nto\n{edge[1][0]}{edge[1][1]}{edge[1][2]}" for edge in G.edges()}
G.edge_cols = {edge: 'blue' for edge in G.edges()}
# Make specific transitions visually distinct if desired
G.edge_cols[((1,0,0), (0,1,0))] = 'green'
G.edge_cols[((2,0,0), (1,1,0))] = 'green'
G.edge_cols[((1,1,0), (0,2,0))] = 'green'
G.edge_cols[((0,1,0), (0,0,1))] = 'purple'
G.edge_cols[((1,1,0), (1,0,1))] = 'purple'


# Draw the graph with the updated logic
draw_graph_updated(G, node_size=2000, rad=-0.15, scale_x=2.0, scale_y=2.0, k_offset_x=0.4, k_offset_y=0.4)