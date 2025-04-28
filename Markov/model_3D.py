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
# from .graph import draw_graph, draw_graph_new
# from graph import draw_graph, draw_graph_new


class MarkovModel():
    
    def __init__(self, config: dict, verbose: bool = False):
        self._verbose = verbose
        # print(config)
        self._lam = config["lam"]
        self._mu = config["mu"]
        self._alpha = config["alpha"]
        self._beta = config["beta"]
        self._theta = config["theta"]
        # self._segments_per_video = config["segments_per_video"]
        # self._preload_segments_per_video = config["preload_segments_per_video"]
        self._max_queue = config["max_queue"]

        # self._preload_videos = self._max_preload_segments / self._preload_segments_per_video
        # if self._preload_videos % 1 != 0:
        #     raise Exception(f"Invalid number of preloaded videos: {self._preload_videos}")
        # assert self._preload_segments_per_video <= self._segments_per_video
        
        # self._lam_factor = config["lam_factor"]
        self._G = self.build_graph()
        if self._verbose:
            print("Markov: Computing probabilities...")
        self.compute_state_probabilities()

    def build_graph(self, color_lam = "blue", color_mu = "black", color_alpha = "green", color_beta = "red", color_theta = "brown"):
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
        
        waiting = [(0,0,0)]
        visited = []
        string_lambda = '$\lambda$'
        # string_mu = "$\mu$"
        string_alpha = "$\\alpha$"
        string_beta = "$\\beta$"
        string_theta = "$\\theta$"
        # string_mix = "$\mu + \\theta$"

        while len(waiting)>0:
            next_alpha_state = None
            next_beta_state = None
            next_lambda_state = None
            next_mu_state = None
            next_theta_state = None
            # print(f"Waiting: aaaaaa")
            current_state = waiting.pop(0)
            visited.append(current_state)
            
            # IDENTIFY NEXT ARRIVAL STATE
            # preloaded_segments = current_state[0] - (current_state[1] - current_state[2])
            if current_state[0] + current_state[2] < self._max_queue:
                next_lambda_state = (current_state[0]+1, 
                                     current_state[1], 
                                     current_state[2])
            else:
                next_lambda_state = current_state  # loops are not added

            
            # IDENTIFY NEXT CONTAINER SPAWNING STATE IF POSSIBLE 
            if current_state[0] > current_state[1]:
                next_alpha_state = (current_state[0], current_state[1] + 1, current_state[2]) 
                # print("current_state:", current_state)
                # print("next_alpha_state:", next_alpha_state)
            # INDENTIFY NEXT REQUEST BEING SERVED STATE IF POSSIBLE
            if current_state[2] < self._max_queue and current_state[0] > 0 and current_state[1] > 0:
                next_beta_state = (current_state[0] - 1, current_state[1] - 1, current_state[2] + 1)
                # print("current_state:", current_state)
                # print("next_beta_state:", next_beta_state)
            # IDENTIFY END OF SERVING STATE
            if current_state[2] > 0:
                next_mu_state = (current_state[0], current_state[1] + 1, current_state[2] - 1)
                # print("current_state:", current_state)
                # print("next_mu_state:", next_mu_state)
            # INDENTIFY CONTAINER TIMEOUT STATE
            # if current_state[1] > 0 and current_state[0] == 0:
            if current_state[1] > 0:
                next_theta_state = (current_state[0], current_state[1] - 1, current_state[2])
                # print("current_state:", current_state)
                # print("next_theta_state:", next_theta_state)
            
                
            # ADD NEW STATES TO GRAPH
            add_transition(current_state, next_lambda_state, rate=self._lam, string=string_lambda, color=color_lam)
            waiting = add_state(visited, waiting, next_lambda_state)

            if next_alpha_state:
                string_alpha = f"{(current_state[0] - current_state[1])}$\\alpha$"
                # print("current_state:", current_state)
                # print("alpha:", selstring_alpha)
                add_transition(current_state, next_alpha_state, rate=(current_state[0] - current_state[1])*self._alpha, string=string_alpha, color=color_alpha)
                waiting = add_state(visited, waiting, next_alpha_state)
            if next_beta_state:
                coe = min(current_state[0], current_state[1])
                string_beta = f"{coe}$\\beta$"
                add_transition(current_state, next_beta_state, rate=self._beta, string=string_beta, color=color_beta)
                add_transition(current_state, next_beta_state, rate=coe*self._beta, string=string_beta, color=color_beta)
                waiting = add_state(visited, waiting, next_beta_state)        
            if next_theta_state:
                string_theta = f"{current_state[1]}$\\theta$"
                add_transition(current_state, next_theta_state, rate=current_state[1]*self._theta, string=string_theta, color=color_theta)
                waiting = add_state(visited, waiting, next_theta_state)
            if next_mu_state:
                string_mu = f"{current_state[2]}$\\mu$"
                add_transition(current_state, next_mu_state, rate=current_state[2]*self._mu, string=string_mu, color=color_mu)
                waiting = add_state(visited, waiting, next_mu_state)        
            # ADD NEW STATES TO QUEUE
                
        # store the parameters in the graph for drawing
        # print(G.nodes)
        G.labels = list(G.nodes)
        G.edge_labels = edge_labels
        G.edge_cols = edge_cols
        G.max_queue = self._max_queue
        G.lam = self._lam
        G.mu = self._mu
        G.theta = self._theta
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
        matrix = np.zeros((self._max_queue+1, self._max_queue+1, self._max_queue+1))
        matrix[0,0,0] = X[ n2i[0,0,0] ]
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
    
    
    # def _get_stalling_states(self):
    #     stalling_states = []
    #     for s in self._n2i:
    #         if np.sum(s)!=0:
    #             if s[0] == 0 or s[1]==s[2]:
    #                 stalling_states.append(s)
    #     return stalling_states
    
    def _get_request_waiting_states(self):
        request_waiting_states = []
        for s in self._n2i:
            if np.sum(s)!=0:
                if s[0] > 0:
                    request_waiting_states.append(s)
                    # print("REQUEST WAITING STATE", s)
                    # print("REQUEST WAITING STATE PROB", self._state_probabilities[s])
        return request_waiting_states
    
    def verify_total_probability(self):
        total = 0
        for s in self._n2i:
            total += self._state_probabilities[s]
            print("REQUEST STATE", s)
            print("REQUEST STATE PROB", self._state_probabilities[s])
        print(f"Total probability across ALL states: {total}")
        return total

    def _get_blocking_states(self):
        blocking_states = []
        for s in self._n2i:
            if s[0] + s[2] == self._max_queue:
                blocking_states.append(s)
        return blocking_states

    def _compute_blocking_ratio(self):
        return np.sum([self._state_probabilities[s] for s in self._get_blocking_states()])
    
    def _compute_waiting_requests(self):
        return np.sum([s[0]*self._state_probabilities[s] for s in self._get_request_waiting_states()])
    # def _get_initial_delay_states(self):
    #     return [(0,0,0)]
    
    def _compute_processing_requests(self):
        return np.sum([s[2]*self._state_probabilities[s] for s in self._get_request_waiting_states()])
    
    def _compute_effective_arrival_rate(self, block_ratio):
        return self._lam*( 1 - block_ratio)
    
    def _compute_waiting_time(self, waiting_requests, effective_arrival_rate):
        # apply Little Law here
        return waiting_requests/effective_arrival_rate
    
    # def _compute_stalling_ratio(self):
    #     return np.sum([self._state_probabilities[s] for s in self._get_stalling_states()])
        
    # def _compute_average_stalling_duration(self):
    #     return 1/(self._lam*self._lam_factor + self._theta)
        
    def _compute_stalling_frequency(self):
        # sum of all state probabilities transitioning to stalling state with playout rate mu
        stalling_states = self._get_stalling_states()
        # print("STALLING STATES", stalling_states)
        reaching_states = set([(u, v, self._G[u][v]['weight']) for node in stalling_states for u, v in self._G.in_edges(node)])
        # print(reaching_states)
        playout_states = []
        swiping_states = []
        for src, dst, weight in reaching_states:
            if src in stalling_states:
                continue
            
            if "mu" in self._G.edge_labels[(src,dst)]:
                playout_states.append(src)
            
            if "theta" in self._G.edge_labels[(src,dst)]:
                swiping_states.append(src)
        
        # sum of all state probabilities transitioning to stalling state
        playout_to_stalling_prob = np.sum([self._state_probabilities[s] for s in playout_states])
        swipe_to_stalling_prob = np.sum([self._state_probabilities[s] for s in swiping_states])
        return playout_to_stalling_prob * self._mu + swipe_to_stalling_prob * self._theta
    
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
    # def _compute_initial_delay_ratio(self):
    #     # sum of all state probabilities transitioning to stalling state with playout rate mu
    #     return self._state_probabilities[0,0,0]
    
    # def _compute_initial_delay_frequency(self):
    #     stalling_states = self._get_initial_delay_states()
    #     reaching_states = set([(u, v, self._G[u][v]['weight']) for node in stalling_states for u, v in self._G.in_edges(node)])
    #     playout_states = []
    #     swiping_states = []
    #     for src, dst, weight in reaching_states:
    #         if src in stalling_states:
    #             continue
            
    #         if "mu" in self._G.edge_labels[(src,dst)]:
    #             playout_states.append(src)
            
    #         if "theta" in self._G.edge_labels[(src,dst)]:
    #             swiping_states.append(src)
        
    #     # sum of all state probabilities transitioning to stalling state
    #     playout_to_stalling_prob = np.sum([self._state_probabilities[s] for s in playout_states])
    #     swipe_to_stalling_prob = np.sum([self._state_probabilities[s] for s in swiping_states])
    #     return playout_to_stalling_prob * self._mu + swipe_to_stalling_prob * self._theta
    
    #TODO: There is a high error for scenarios where ratios are close to 0
    # def _compute_average_initial_delay_duration(self):
    #     return 1/(self._lam * self._lam_factor)
    
    def _compute_wasted_segments(self):
        expected_wasted_segments = []
        for state in self._n2i.keys():
            wasted_segments = state[1] - state[2]
            expected_wasted_segments.append(self._state_probabilities[state] * self._theta * wasted_segments)
        return np.sum(expected_wasted_segments)
    
    def _compute_mean_wasted_segments(self):
        return self._segments_per_video * self._compute_wastage_ratio()
    
    def _compute_played_segments(self):
        expected_played_segments = []
        for state in self._n2i.keys():
            if state[0] > 0 and state[1]!=state[2]:
                expected_played_segments.append(self._state_probabilities[state] * self._mu)
        return np.sum(expected_played_segments)
    
    def _compute_mean_played_segments(self):
        return self._segments_per_video * self._compute_play_ratio()
    
    def _compute_downloaded_segments(self):
        expected_downloaded_segments = []
        for state in self._n2i.keys():
            preloaded_segments = state[0] - (state[1] - state[2])
            # Add downloaded segments only if there is an arrival in the corresponding state
            # Happens only when preloaded segments are unequal to maximum preloadable segments
            if preloaded_segments != self._max_preload_segments:
                # Download with faster rate if in low-buffer state
                if state[0] < self._segments_per_video:
                    expected_downloaded_segments.append(self._state_probabilities[state] * (self._lam*self._lam_factor))
                else:
                    expected_downloaded_segments.append(self._state_probabilities[state] * self._lam)
        return np.sum(expected_downloaded_segments)
        
    def _compute_wastage_ratio(self):
        return self._compute_wasted_segments()/self._compute_downloaded_segments()
    
    def _compute_play_ratio(self):
        return self._compute_played_segments()/self._compute_downloaded_segments()
    
    def _compute_buffered_segments(self):
        s = np.sum(self._state_probabilities, axis=(1,2))
        return s @ np.arange(len(s))

    def _compute_downloaded_segments(self):
        s = np.sum(self._state_probabilities, axis=(0,2))
        return s @ np.arange(len(s))
    
    def _compute_played_segments(self):
        s = np.sum(self._state_probabilities, axis=(0,1))
        return s @ np.arange(len(s))
    
    def get_metrics(self):
        if self._verbose:
            print("Markov: Collecting metrics...")
        
        block_ratio = self._compute_blocking_ratio()
        waiting_requests = self._compute_waiting_requests()
        processing_requests = self._compute_processing_requests()
        effective_arrival_rate = self._compute_effective_arrival_rate(block_ratio)
        waiting_time = self._compute_waiting_time(waiting_requests, effective_arrival_rate)
        self.verify_total_probability()
        # latency = waiting_time + 

        return {
            "blocking_ratios": [block_ratio],
            "waiting_requests": [waiting_requests],
            "processing_requests": [processing_requests],
            "effective_arrival_rates": [effective_arrival_rate],
            "waiting_times": [waiting_time],
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
    
    
if __name__=="__main__":
    config = {
        "lam": 5,
        "mu": 2,
        "alpha": 1/5,
        "beta": 1000,
        "theta": 1/2,
        # "segments_per_video": 3,
        # "preload_segments_per_video": 1,
        "max_queue": 4, # queue
        "serving_time": "exponential",
        "arrivals": "exponential",
        # "lam_factor": 1,
        "simulation_duration": 1000,
        "num_runs": 1
    }
    m = MarkovModel(config, verbose=False)
    G = m._G
    # draw_graph_new(G, node_size=1000)
    print(m.get_metrics())
