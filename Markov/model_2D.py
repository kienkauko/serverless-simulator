#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 20:20:26 2025

@author: https://hossfeld.github.io/performance-modeling/chapter4_markovianSystems/ch4-graph2MarkovModel.html
"""

import networkx as nx  # For the magic
import matplotlib.pyplot as plt  # For plotting
import numpy as np
from scipy import linalg
import math
# from graph import draw_graph_updated


class MarkovModel():
    
    def __init__(self, config: dict, verbose: bool = False):
        self._verbose = verbose
        # print(config)
        self._lam = config["lam"]
        self._mu = config["mu"]
        self._alpha = 1/(1/config["spawn_rate"] + 1/config["mu"])
        print(f"Alpha: {self._alpha}")
        self._spawn_rate = config["spawn_rate"]
        self._max_queue_warm = config["queue_warm"]
        self._max_queue_cold = config["queue_cold"]
        self._ram_warm = config["ram_warm"]
        self._cpu_warm = config["cpu_warm"]
        self._ram_active = config["ram_demand"]
        self._cpu_active = config["cpu_demand"]
        self._peak_power = config["peak_power"]
        self._power_scale = config["power_scale"]
        # self._preload_videos = self._max_preload_segments / self._preload_segments_per_video
        # if self._preload_videos % 1 != 0:
        #     raise Exception(f"Invalid number of preloaded videos: {self._preload_videos}")
        # assert self._preload_segments_per_video <= self._segments_per_video
        
        # self._lam_factor = config["lam_factor"]
        self._G = self.build_graph()
        if self._verbose:
            print("Markov: Computing probabilities...")
        self.compute_state_probabilities()

    def build_graph(self, color_lam = "blue", color_mu = "black", color_alpha = "green"):
        G = nx.DiGraph()
        edge_labels={}
        edge_cols = {}
        
        # Function to contruct transition between two states
        def add_transition(origin_state, destination_state, rate, string='rate', color=plt.cm.tab10(5)):
            if origin_state == destination_state:
                return
            G.add_edge(origin_state, destination_state, weight=rate, label=string)
            edge_labels[(origin_state, destination_state)] = string # label=string
            edge_cols[(origin_state, destination_state)] = color
        # Function to add a state
        def add_state(visited, waiting, state):
            if state in visited or state in waiting:
                return waiting
            waiting.append(state)
            return waiting
        
        waiting = [(0,0)]
        visited = []
        string_lambda = '$\lambda$'
        string_mu = "$\mu$"
        string_alpha = "$\\alpha$"

        while len(waiting)>0:
            next_alpha_state = None
            next_lambda_state = None
            next_mu_state = None
            current_state = waiting.pop(0)
            visited.append(current_state)
            
            # IDENTIFY NEXT Warm state
            if current_state[0]  < self._max_queue_warm:
                next_lambda_state = (current_state[0]+1, current_state[1])
            else:
                next_lambda_state = current_state  # loops are not added

            # IDENTIFY NEXT Cold state
            if current_state[1] < self._max_queue_cold and current_state[0] == self._max_queue_warm:
                next_lambda_state = (current_state[0], current_state[1] + 1)
            
            # IDENTIFY next alpha state (cold removal)
            if current_state[1] > 0:
                next_alpha_state = (current_state[0], current_state[1] - 1)
            
            # Identify next mu state (warm removal)
            if current_state[0] > 0:
                next_mu_state = (current_state[0] - 1, current_state[1])
                
            # ADD NEW STATES TO GRAPH
            if next_lambda_state:
                add_transition(current_state, next_lambda_state, rate=self._lam, string=string_lambda, color=color_lam)
                waiting = add_state(visited, waiting, next_lambda_state)

            if next_alpha_state:
                string_alpha = f"{current_state[1]}$\\alpha$"
                add_transition(current_state, next_alpha_state, rate=current_state[1]*self._alpha, string=string_alpha, color=color_alpha)
                waiting = add_state(visited, waiting, next_alpha_state)

            if next_mu_state:
                string_mu = f"{current_state[0]}$\\mu$"
                add_transition(current_state, next_mu_state, rate=current_state[0]*self._mu, string=string_mu, color=color_mu)
                waiting = add_state(visited, waiting, next_mu_state)        
            # ADD NEW STATES TO QUEUE
                
        # store the parameters in the graph for drawing
        # print(G.nodes)
        G.labels = list(G.nodes)
        G.edge_labels = edge_labels
        G.edge_cols = edge_cols
        G.max_queue = max(self._max_queue_warm, self._max_queue_cold)
        G.alpha = self._alpha
        G.lam = self._lam
        G.mu = self._mu
        
        return G
    
    def compute_state_probabilities(self):
        # compute transition matrix
        Q2, n2i = self.create_rate_matrix()

        if self._verbose:
            print(f'Q=\n{Q2}\n')
        
        Q2[:, -1] = 1
        if self._verbose:        
            print(f'Matrix is changed to\nQ2=\n{Q2}\n')
        
        b = np.zeros(len(Q2))
        b[-1] = 1
        if self._verbose:
            print(f'b=\n{b}\n')
        
        # state probabilities
        X = b @ linalg.inv(Q2) # compute the matrix inverse
        if self._verbose:
            print(f'X=\n{X}\n')
        
        # Generate a matrix with P(X,Y)
        # max_segments = self._segments_per_video + self._max_preload_segments
        matrix = np.zeros((self._max_queue_warm + 1, self._max_queue_cold + 1))
        matrix[0,0] = X[ n2i[0,0] ]
        for s in list(n2i.keys()):
            matrix[s] = X[ n2i[s] ] 
    
        self._state_probabilities = matrix
        self._n2i = n2i
        
    def create_rate_matrix(self):
        """
        Draws the the state transition diagram described by the NetworkX graph G.
        
        Args:
            G (networkx.DiGraph): State transition graph.
            
        Returns:
            Q (np.array): State transition matrix used for computing system characteristics.
            n2i (dict): Mapping between the system states and the index of the rate matrix. The keys are the system states (x,y).
        """      
        n2i = {} # mapping between system states (x,y) and the index i of the matrix
        nodes = sorted(list(self._G.nodes))
        for i,node in enumerate(nodes):
            n2i[node] = i         
        Q = np.zeros((len(nodes),len(nodes))) # rate matrix
        
        for edge in self._G.edges:
            i0 = n2i[edge[0]]
            i1 = n2i[edge[1]]
            Q[i0,i1] = self._G[edge[0]][edge[1]]["weight"] 
            
        np.fill_diagonal(Q, -Q.sum(axis=1))
        return Q, n2i
    
    def get_graph(self):
        return self._G
        
    def _get_state_probabilities(self):
        return self._state_probabilities
    
    
    def _get_request_waiting_states(self):
        request_waiting_states = []
        for s in self._n2i:
            if np.sum(s)!=0:
                if s[0] > 0:
                    request_waiting_states.append(s)
                    # print("REQUEST WAITING STATE", s)
                    # print("REQUEST WAITING STATE PROB", self._state_probabilities[s])
        return request_waiting_states
    def _get_request_processing_states(self):
        request_waiting_states = []
        for s in self._n2i:
            if np.sum(s)!=0:
                if s[1] > 0:
                    request_waiting_states.append(s)
                    # print("REQUEST WAITING STATE", s)
                    # print("REQUEST WAITING STATE PROB", self._state_probabilities[s])
        return request_waiting_states
    
    def verify_total_probability(self):
        total = 0
        for s in self._n2i:
            total += self._state_probabilities[s]
            print("REQUEST STATE", s)
            # print("REQUEST STATE PROB", self._state_probabilities[s])
        print(f"Total probability across ALL states: {total}")
        return total

    def _get_blocking_states(self):
        blocking_states = []
        for s in self._n2i:
            if s[0] == self._max_queue_warm and s[1] == self._max_queue_cold:
                blocking_states.append(s)
        print("BLOCKING STATE", s)
        return blocking_states

    def _compute_blocking_ratio(self):
        return np.sum([self._state_probabilities[s] for s in self._get_blocking_states()])
    
    def _compute_requests_in_system(self):
        return np.sum([(s[0] + s[1])*self._state_probabilities[s] for s in self._n2i])
    
    def _compute_processing_requests(self):
        return np.sum([s[1]*self._state_probabilities[s] for s in self._n2i])
    
    def _compute_idling_requests(self):
        return np.sum([s[2]*self._state_probabilities[s] for s in self._n2i])
    
    def _compute_warm_job(self):
        return np.sum([s[0]*self._state_probabilities[s] for s in self._n2i])
    
    def _compute_cold_job(self):
        return np.sum([s[1]*self._state_probabilities[s] for s in self._n2i])
    
    def _compute_resource_usage(self, resource):
        if resource == "cpu":
            active = self._cpu_active
            warm = self._cpu_warm
            transit = 3.0
        elif resource == "ram":
            active = self._ram_active
            warm = self._ram_warm
            transit = warm
        else:
            raise ValueError("Resource must be either 'cpu' or 'ram'")
        
        spawn_time = 1.0 / self._spawn_rate  # Derive spawn_rate from alpha
        serving_time = 1.0 / self._mu

        mean_cold_consume = (spawn_time*transit + serving_time*active)/(spawn_time + serving_time)

        resource_usage = self._max_queue_warm*warm + self._compute_warm_job()*(active - warm) \
            + self._compute_cold_job()*mean_cold_consume
        
        return resource_usage

    def _computer_power_usage(self, cpu_usage):

        driven_resource = max(self._cpu_active, self._ram_active)
        num_con_per_server = math.floor(100/ driven_resource)  # Assume 100% resource capacity per server
        num_job = self._max_queue_warm + self._compute_cold_job()
        on_server = math.ceil(num_job/num_con_per_server)
        base_power = self._peak_power * self._power_scale

        return on_server * base_power + (cpu_usage / 100) * (self._peak_power - base_power)


    def _compute_effective_arrival_rate(self, block_ratio):
        return self._lam*( 1 - block_ratio)
    
    def _compute_latency(self, waiting_requests, effective_arrival_rate):
        # apply Little Law here
        return waiting_requests/effective_arrival_rate
    
    def _compute_cpu_usage(self):
        return np.sum([((s[0] + s[2])*self._cpu_warm + s[1]*self._cpu_active)*self._state_probabilities[s] for s in self._n2i.keys()])
    
    def _compute_ram_usage(self):
        return np.sum([((s[0] + s[2])*self._ram_warm + s[1]*self._ram_active)*self._state_probabilities[s] for s in self._n2i.keys()])
    

    def calculate_queue_pgf(self, z=1):
        """
        Calculate the probability generating function (PGF) for the number of requests in the queue.
        
        Q(z) = Σ(π_{i,j,k}) z^i from i=0 to L_q
        
        Where:
        - π_{i,j,k} are the steady state probabilities
        - L_q is max_queue
        - z is a complex number, default z=1
        
        Args:
            z (float): Value at which to evaluate the PGF
        
        Returns:
            float: PGF value at the specified z
        """
        result = 0
        for s in self._n2i:
            i, j, k = s  # Unpack state tuple (i = waiting requests, j = containers, k = processing)
            result += self._state_probabilities[s] * (z ** i)  # z^i where i is the queue length
        return result

    def _compute_mean_queue_length_pgf(self):
        """
        Compute mean queue length by evaluating the first derivative of Q(z) at z=1.
        
        E[queue length] = Q'(1)
        
        We use numerical differentiation with a small delta.
        """
        delta = 0.0001
        # First derivative approximation using central difference
        dQdz_at_1 = (self.calculate_queue_pgf(1 + delta) - self.calculate_queue_pgf(1 - delta)) / (2 * delta)
        return dQdz_at_1

    def _compute_waiting_time_pgf(self):
        """
        Calculate the mean waiting time using the mean queue length derived from PGF
        and applying Little's Law.
        
        W_q = L_q / λ_eff
        """
        mean_queue_length = self._compute_mean_queue_length_pgf()
        block_ratio = self._compute_blocking_ratio()
        effective_arrival_rate = self._compute_effective_arrival_rate(block_ratio)
        
        # Apply Little's Law
        return mean_queue_length / effective_arrival_rate
   
    
    def get_metrics(self):
        if self._verbose:
            print("Markov: Collecting metrics...")
        
        block_ratio = self._compute_blocking_ratio()
        requests_in_system = self._compute_requests_in_system()
        # processing_requests = self._compute_processing_requests()
        # idling_requests = self._compute_idling_requests()
        # total_requests = waiting_requests + processing_requests 
        effective_arrival_rate = self._compute_effective_arrival_rate(block_ratio)
        latency = self._compute_latency(requests_in_system, effective_arrival_rate)
        cpu_usage = self._compute_resource_usage("cpu")
        ram_usage = self._compute_resource_usage("ram")
        power_usage = self._computer_power_usage(cpu_usage)
        # cpu_usage = self._compute_cpu_usage()
        # ram_usage = self._compute_ram_usage()
        # cpu_usage_per_request = cpu_usage/total_requests
        # ram_usage_per_request = ram_usage/total_requests

        return {
            "blocking_ratios": [block_ratio],
            "requests_in_system": [requests_in_system],
            "latency": [latency],
            "cpu_usage": [cpu_usage],
            "ram_usage": [ram_usage],
            "power_usage": [power_usage],
            # "processing_requests": [processing_requests],
            # "idling_requests": [idling_requests],
            # "effective_arrival_rates": [effective_arrival_rate],
            # "waiting_times": [waiting_time],
            # 'mean_cpu_usage': [cpu_usage],
            # 'mean_ram_usage': [ram_usage],
            # 'mean_cpu_usage_per_request': [cpu_usage_per_request],
            # 'mean_ram_usage_per_request': [ram_usage_per_request],
            # "states": [self._get_state_probabilities()]
            # "stalling_ratios": [self._compute_stalling_ratio()],
            # "average_stalling_durations": [self._compute_average_stalling_duration()],
            # "stalling_frequencies": [self._compute_stalling_frequency()],
            # "initial_delay_ratios": [self._compute_initial_delay_ratio()],
            # "average_initial_delay_durations": [self._compute_average_initial_delay_duration()],
            # "initial_delay_frequencies": [self._compute_initial_delay_frequency()],
            # "wastage_ratios": [self._compute_wastage_ratio()],
            # "wastage_mean_segments": [self._compute_mean_wasted_segments()],
            # "buffered_segments": [self._compute_buffered_segments()],
            # "downloaded_segments": [self._compute_downloaded_segments()],
            # "played_segments": [self._compute_played_segments()]
        }
    

def draw_graph_updated(G, node_size=1500, rad=-0.2, scale_x=0.5, scale_y=1.0):
    # --- 1. Calculate Node Positions ---
    pos = {}
    for node in G.nodes():
        if isinstance(node, tuple) and len(node) == 2:
            try:
                # Ensure components are numeric and non-negative
                i, j = map(float, node)  # Convert to float for calculation
                if not (i >= 0 and j >= 0):
                    raise ValueError("Components must be non-negative")
            except (ValueError, TypeError):
                print(f"Warning: Node {node} has invalid or non-numeric components. Skipping positioning.")
                pos[node] = (0, 0)  # Assign default position
                continue
                
            # Position nodes in a 2D grid where:
            # x increases with i (horizontally)
            # y decreases with j (vertically downward)
            x = scale_x * i
            y = -scale_y * j  # Negative to move downward
            pos[node] = (x, y)
        else:
            # Handle nodes not in the expected (i, j) tuple format
            print(f"Warning: Node '{node}' is not in (i, j) format. Placing at (0,0).")
            pos[node] = (0, 0)

    # --- 2. Prepare for Drawing ---
    plt.figure(figsize=(14, 7), clear=True)

    # --- 3. Draw the Graph Components ---
    # Use getattr to safely access attributes that may not exist
    labels = getattr(G, 'labels', {})
    edge_labels = getattr(G, 'edge_labels', {})
    edge_cols = getattr(G, 'edge_cols', {})
    
    # Default color if edge_cols doesn't contain an edge
    default_color = 'black'
    edge_colors = [edge_cols.get(edge, default_color) for edge in G.edges]

    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_shape='o', node_size=node_size)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    edges = nx.draw_networkx_edges(G, pos, width=1, edge_color=edge_colors,
                                 node_size=node_size, 
                                 arrows=True, arrowstyle='-|>',
                                 connectionstyle=f"arc3,rad={rad}")
            
    my_draw_networkx_edge_labels(G, pos, ax=plt.gca(), edge_labels=edge_labels, rotate=False, rad=rad)
    plt.axis('off')
    plt.show()

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


if __name__=="__main__":
    config = {
        "lam": 1,
        "mu": 1,
        "spawn_rate": 1,
        "queue_warm":5, # queue
        "queue_cold": 5, # queue
        "serving_time": "exponential",
        "arrivals": "exponential",
        # "lam_factor": 1,
        "ram_warm": 30,
        "cpu_warm": 1,
        "ram_demand": 40,
        "cpu_demand": 50,
        "peak_power": 150.0,
        "power_scale": 0.5,  # Power scale factor

    }
    m = MarkovModel(config, verbose=False)
    G = m._G
    # draw_graph_updated(G, node_size=1000)
    print(m.get_metrics())
