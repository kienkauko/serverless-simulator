#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 20:52:01 2025

@author: nikolas
"""

import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items

def draw_graph_new (G, node_size=1500, rad=-0.2, scale_x=1.0, scale_y=1.0, k_offset_x=0.5, k_offset_y=0.5):
    """
    Draws the state transition diagram described by the NetworkX graph G.
    States are expected to be tuples (i, j, k) where i, j, k >= 0.
    - Increasing 'i' moves nodes horizontally to the right.
    - Increasing 'j' moves nodes vertically downwards.
    - Increasing 'k' offsets nodes slightly diagonally (default: up-right)
      relative to their (i, j) position to avoid overlap.

    Args:
        G (networkx.DiGraph): State transition graph. Nodes MUST be tuples (i, j, k).
                              Assumes G might have attributes G.labels, G.edge_labels, G.edge_cols.
                              If not present, defaults will be used.
        node_size (int): Size of a node in the diagram.
        rad (float): Curvature of the edges (for arc3 connectionstyle).
        scale_x (float): Base spacing between nodes along the i-axis (horizontal).
        scale_y (float): Base spacing between nodes along the j-axis (vertical).
        k_offset_x (float): Horizontal offset added per unit increase in k.
        k_offset_y (float): Vertical offset added per unit increase in k.
                           Positive value offsets slightly 'up' relative to the base -j position.
    """

    # --- 1. Calculate Node Positions ---
    pos = {}
    max_queue = G.max_queue
    for node in G.nodes():
        if isinstance(node, tuple) and len(node) == 3:
            i, j, k = node
            if not (isinstance(i, (int, float)) and isinstance(j, (int, float)) and isinstance(k, (int, float)) and i >= 0 and j >= 0 and k >= 0):
                 print(f"Warning: Node {node} has invalid components. Skipping positioning.")
                 continue # Skip nodes with invalid i,j,k

            # Base position determined by i (x) and j (y, negative for downward)
            base_x = scale_x * i
            base_y = -scale_y * j # Negative j moves nodes down

            # Apply offset based on k to avoid overlaps
            # The offset moves nodes slightly diagonally (default up and right)
            final_x = base_x + k * k_offset_x
            final_y = base_y + k * k_offset_y

            pos[node] = (final_x, final_y)
        else:
            # Handle nodes not in the expected (i, j, k) tuple format
            print(f"Warning: Node '{node}' is not in (i, j, k) format. Placing at (0,0).")
            pos[node] = (0, 0) # Assign a default position

    # --- 2. Prepare for Drawing ---
    plt.figure(figsize=(14, 7), clear=True)

    # --- 3. Draw the Graph Components ---
    labels = G.labels
    edge_labels = G.edge_labels 
    edge_cols = G.edge_cols
    

    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_shape ='o', node_size=node_size)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')  # Draw node labels
    edges = nx.draw_networkx_edges(G, pos,  width=1, edge_color=[edge_cols[edge] for edge in G.edges], 
                           node_size = node_size, 
                           arrows = True, arrowstyle = '-|>',
                           connectionstyle=f"arc3,rad={rad}")
            
    #nx.draw_networkx_edge_labels(G, pos, edge_labels, label_pos=0.4, verticalalignment='top', horizontalalignment="center")
    my_draw_networkx_edge_labels(G, pos, ax=plt.gca(), edge_labels=edge_labels,rotate=False, rad = rad)
    plt.axis('off')
    plt.show()



def draw_graph(G, node_size=1500, rad=-0.2):     
    """
    Draws the the state transition diagram described by the NetworkX graph G.
    
    Args:
        G (networkx.DiGraph): State transition graph.
        node_size (int): Size of a node in the diagram. 
        rad (float): Degree of bended arrows.        
    """      
    labels = G.labels
    edge_labels = G.edge_labels 
    edge_cols = G.edge_cols
    max_queue = G.max_queue
    plt.figure(figsize=(14,7), clear=True)
    pos = {state:list(state) for state in G.nodes}
    # print(pos)
    spaces_x = np.linspace(-0.5, 0.5, num=max_queue+1)
    spaces_y = np.linspace(0, 1.0, num=max_queue+1)
    offsets_x = {i: spaces_x[i] for i in range(len(spaces_x))}
    offsets_y = {i: spaces_y[i] for i in range(len(spaces_y))}
    for k, v in pos.items():
        if len(v) >= 3:
            if v[2] > 0:
                pos[k] = [v[0] + (1/2)*offsets_y[v[2]], -((1/4)*v[2] + offsets_x[v[1]])]
            else:
                pos[k] = [v[0], -(offsets[v[1]])]
            print(f'pos[{k}] = {pos[k]}')
        else:
            break
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_shape ='o', node_size=node_size)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')  # Draw node labels
    edges = nx.draw_networkx_edges(G, pos,  width=1, edge_color=[edge_cols[edge] for edge in G.edges], 
                           node_size = node_size, 
                           arrows = True, arrowstyle = '-|>',
                           connectionstyle=f"arc3,rad={rad}")
            
    #nx.draw_networkx_edge_labels(G, pos, edge_labels, label_pos=0.4, verticalalignment='top', horizontalalignment="center")
    my_draw_networkx_edge_labels(G, pos, ax=plt.gca(), edge_labels=edge_labels,rotate=False, rad = rad)
    plt.axis('off')
    plt.show()