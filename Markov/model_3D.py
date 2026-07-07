#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D CTMC for the dynamic (idle-timeout) serverless pool.

State  s = (x1, x2, x3) in N_0^3:
    x1 : containers in the cold-start (spawning) phase
    x2 : containers actively processing requests
    x3 : warm (idle) containers

Transitions
-----------
Arrival (rate lambda):
    * warm hit  (x3 > 0)               -> (x1,   x2+1, x3-1)
    * cold start(x3 = 0, x1+x2 < n)    -> (x1+1, x2,   x3  )
    * blocked   (x3 = 0, x1+x2 = n)    -> no transition (request rejected)
Spawn completion (rate x1 * alpha)     -> (x1-1, x2+1, x3  )
Processing completion (rate x2 * mu)   -> (x1,   x2-1, x3+1)
Warm timeout (rate x3 * theta)         -> (x1,   x2,   x3-1)

The system provides n = max_queue container slots in total, so
x1 + x2 + x3 <= n is preserved by every transition and blocking occurs
exactly when x1 + x2 = n (which implies x3 = 0).

Based on
@author: https://hossfeld.github.io/performance-modeling/chapter4_markovianSystems/ch4-graph2MarkovModel.html
"""

import networkx as nx  # For the magic
import matplotlib.pyplot as plt  # For plotting
import numpy as np
import math  # For mathematical operations
from scipy import linalg
# from graph import draw_graph_updated


class MarkovModel():

    def __init__(self, config: dict, verbose: bool = False):
        self._verbose = verbose
        # print(config)
        self._lam = config["arrival_rate"]
        self._mu = config["service_rate"]            # processing completion rate (mu = 1/E[B_w])
        self._spawn_rate = config["spawn_rate"]
        self._alpha = config["spawn_rate"]           # spawn completion rate (cold -> active)
        self._theta = config["theta"]                # warm container timeout rate
        self._max_queue = config["max_queue"]        # n: total number of container slots
        self._ram_warm = config["ram_warm"]
        self._cpu_warm = config["cpu_warm"]
        self._ram_active = config["ram_demand"]
        self._cpu_active = config["cpu_demand"]
        # Cold-start (transit) resource demand held by a spawning container (X1).
        # Falls back to the warm demand when not provided.
        self._cpu_transit = config.get("cpu_transit", config["cpu_warm"])
        self._ram_transit = config.get("ram_transit", config["ram_warm"])
        self._power_max = config["power_max"]
        self._power_min_scale = config["power_min_scale"]
        self._G = self.build_graph()
        if self._verbose:
            print("Markov: Computing probabilities...")
        self.compute_state_probabilities()

    def build_graph(self, color_lam="blue", color_mu="black", color_alpha="green", color_theta="brown"):
        G = nx.DiGraph()
        edge_labels = {}
        edge_cols = {}

        # Function to construct transition between two states
        def add_transition(origin_state, destination_state, rate, string='rate', color=plt.cm.tab10(5)):
            if origin_state == destination_state:
                return
            G.add_edge(origin_state, destination_state, weight=rate, label=string)
            edge_labels[(origin_state, destination_state)] = string  # label=string
            edge_cols[(origin_state, destination_state)] = color

        # Function to add a state
        def add_state(visited, waiting, state):
            if state in visited or state in waiting:
                return waiting
            waiting.append(state)
            return waiting

        waiting = [(0, 0, 0)]
        visited = []
        string_lambda = r'$\lambda$'
        string_mu = r"$\mu$"
        string_alpha = r"$\alpha$"
        string_theta = r"$\theta$"

        while len(waiting) > 0:
            next_alpha_state = None
            next_mu_state = None
            next_lambda_state = None
            next_theta_state = None
            current_state = waiting.pop(0)
            visited.append(current_state)
            x1, x2, x3 = current_state

            # IDENTIFY next arrival state
            if x3 > 0:
                # warm container available -> turn it into an active one
                next_lambda_state = (x1, x2 + 1, x3 - 1)
            elif x1 + x2 < self._max_queue:
                # no warm container, but a free slot -> cold start
                next_lambda_state = (x1 + 1, x2, x3)
            else:
                # blocking state: x1 + x2 == n (and x3 == 0) -> request rejected
                next_lambda_state = current_state  # self-loops are not added

            # IDENTIFY spawn completion state (cold -> active)
            if x1 > 0:
                next_alpha_state = (x1 - 1, x2 + 1, x3)

            # IDENTIFY processing completion state (active -> warm)
            if x2 > 0:
                next_mu_state = (x1, x2 - 1, x3 + 1)

            # IDENTIFY warm timeout state (warm -> null)
            if x3 > 0:
                next_theta_state = (x1, x2, x3 - 1)

            # ADD NEW STATES TO GRAPH
            add_transition(current_state, next_lambda_state, rate=self._lam, string=string_lambda, color=color_lam)
            waiting = add_state(visited, waiting, next_lambda_state)

            if next_alpha_state:
                string_alpha = f"{x1}$\\alpha$"
                add_transition(current_state, next_alpha_state, rate=x1 * self._alpha, string=string_alpha, color=color_alpha)
                waiting = add_state(visited, waiting, next_alpha_state)

            if next_mu_state:
                string_mu = f"{x2}$\\mu$"
                add_transition(current_state, next_mu_state, rate=x2 * self._mu, string=string_mu, color=color_mu)
                waiting = add_state(visited, waiting, next_mu_state)

            if next_theta_state:
                string_theta = f"{x3}$\\theta$"
                add_transition(current_state, next_theta_state, rate=x3 * self._theta, string=string_theta, color=color_theta)
                waiting = add_state(visited, waiting, next_theta_state)

        # store the parameters in the graph for drawing
        G.labels = list(G.nodes)
        G.edge_labels = edge_labels
        G.edge_cols = edge_cols

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
        X = b @ linalg.inv(Q2)  # compute the matrix inverse
        if self._verbose:
            print(f'X=\n{X}\n')

        # Generate a matrix with P(x1, x2, x3)
        matrix = np.zeros((self._max_queue + 1, self._max_queue + 1, self._max_queue + 1))
        matrix[0, 0, 0] = X[n2i[0, 0, 0]]
        for s in list(n2i.keys()):
            matrix[s] = X[n2i[s]]

        self._state_probabilities = matrix
        self._n2i = n2i

    def create_rate_matrix(self):
        """
        Builds the state transition (rate) matrix from the NetworkX graph G.

        Returns:
            Q (np.array): State transition matrix used for computing system characteristics.
            n2i (dict): Mapping between the system states (x1, x2, x3) and the index of the rate matrix.
        """
        n2i = {}  # mapping between system states (x1, x2, x3) and the index i of the matrix
        nodes = sorted(list(self._G.nodes))
        for i, node in enumerate(nodes):
            n2i[node] = i
        Q = np.zeros((len(nodes), len(nodes)))  # rate matrix

        for edge in self._G.edges:
            i0 = n2i[edge[0]]
            i1 = n2i[edge[1]]
            Q[i0, i1] = self._G[edge[0]][edge[1]]["weight"]

        np.fill_diagonal(Q, -Q.sum(axis=1))
        return Q, n2i

    def get_graph(self):
        return self._G

    def _get_state_probabilities(self):
        return self._state_probabilities

    def verify_total_probability(self):
        total = 0
        for s in self._n2i:
            total += self._state_probabilities[s]
            print("STATE", s)
        print(f"Total probability across ALL states: {total}")
        return total

    # ------------------------------------------------------------------ #
    #  Aggregate state quantities                                        #
    # ------------------------------------------------------------------ #
    def _get_blocking_states(self):
        # blocked when spawning + processing fill all slots (x1 + x2 == n)
        blocking_states = []
        for s in self._n2i:
            if s[0] + s[1] == self._max_queue:
                blocking_states.append(s)
        return blocking_states

    def _compute_blocking_ratio(self):
        return np.sum([self._state_probabilities[s] for s in self._get_blocking_states()])

    def _compute_cold_requests(self):
        # mean number of containers in the cold-start (spawning) phase, E[X1]
        return np.sum([s[0] * self._state_probabilities[s] for s in self._n2i])

    def _compute_active_requests(self):
        # mean number of actively processing containers, E[X2]
        return np.sum([s[1] * self._state_probabilities[s] for s in self._n2i])

    def _compute_idling_container(self):
        # mean number of warm (idle) containers, E[X3]
        return np.sum([s[2] * self._state_probabilities[s] for s in self._n2i])

    def _compute_requests_in_system(self):
        # requests being served = spawning + processing (warm idle hold no request)
        return np.sum([(s[0] + s[1]) * self._state_probabilities[s] for s in self._n2i])

    def _compute_effective_arrival_rate(self, block_ratio):
        return self._lam * (1 - block_ratio)

    def _compute_waiting_time(self, requests_in_system, effective_arrival_rate):
        # apply Little's Law here
        return requests_in_system / effective_arrival_rate

    def _compute_warm_hit_ratio(self):
        """
        Probability that an arriving request finds a warm container (warm hit).
        By PASTA this is P(X3 > 0) among admitted arrivals.
        """
        p_b = self._compute_blocking_ratio()
        p_warm = np.sum([self._state_probabilities[s] for s in self._n2i if s[2] > 0])
        if p_b < 1.0:
            return p_warm / (1.0 - p_b)
        return 0.0

    # ------------------------------------------------------------------ #
    #  Resource & power usage                                            #
    # ------------------------------------------------------------------ #
    def _compute_resource_usage(self, resource: str):
        """
        Mean resource consumption.
            * spawning containers (X1) hold the cold-start (transit) demand,
            * active containers (X2) hold the active demand,
            * warm idle containers (X3) hold the warm (baseline) demand.
        """
        if resource == 'cpu':
            warm = self._cpu_warm
            active = self._cpu_active
            transit = self._cpu_transit
        elif resource == 'ram':
            warm = self._ram_warm
            active = self._ram_active
            transit = self._ram_transit
        else:
            raise ValueError("Resource must be either 'cpu' or 'ram'")

        cold_containers = self._compute_cold_requests()
        active_containers = self._compute_active_requests()
        idle_containers = self._compute_idling_container()

        cold_consumption = cold_containers * transit
        active_consumption = active_containers * active
        idle_consumption = idle_containers * warm
        return cold_consumption + active_consumption + idle_consumption

    def _compute_power_usage(self, cpu_usage):
        """
        Mean power across the (time-averaged) number of ON servers.
            P = n_on * P_base + (cpu_usage / 100) * (P_max - P_base)
        where n_on packs the mean container count into servers and
        P_base = P_max * power_min_scale. Mirrors the simulator's per-server
        idle-floor-plus-CPU-proportional power model.
        """
        total_container = (self._compute_cold_requests()
                           + self._compute_active_requests()
                           + self._compute_idling_container())
        driving_resource = max(self._cpu_active, self._ram_active)
        num_con_per_server = math.floor(100 / driving_resource)
        num_ON_server = (math.ceil(total_container / num_con_per_server)
                         if num_con_per_server > 0 else 0)
        if self._verbose:
            print(f"Total container: {total_container}, ON servers: {num_ON_server}")
        power_min = self._power_max * self._power_min_scale
        return num_ON_server * power_min + (cpu_usage / 100) * (self._power_max - power_min)

    # ------------------------------------------------------------------ #
    #  Queue PGF utilities                                               #
    # ------------------------------------------------------------------ #
    def calculate_queue_pgf(self, z=1):
        """
        PGF for the number of requests in system (spawning + processing).

        Q(z) = Σ π_{x1,x2,x3} z^{x1+x2}
        """
        result = 0
        for s in self._n2i:
            x1, x2, x3 = s
            result += self._state_probabilities[s] * (z ** (x1 + x2))
        return result

    def _compute_mean_queue_length_pgf(self):
        """E[requests in system] = Q'(1), via central difference."""
        delta = 0.0001
        dQdz_at_1 = (self.calculate_queue_pgf(1 + delta) - self.calculate_queue_pgf(1 - delta)) / (2 * delta)
        return dQdz_at_1

    def _compute_waiting_time_pgf(self):
        """W = L / lambda_eff (Little's Law) using the PGF-derived mean."""
        mean_queue_length = self._compute_mean_queue_length_pgf()
        block_ratio = self._compute_blocking_ratio()
        effective_arrival_rate = self._compute_effective_arrival_rate(block_ratio)
        return mean_queue_length / effective_arrival_rate

    # ------------------------------------------------------------------ #
    #  Metrics collection                                                #
    # ------------------------------------------------------------------ #
    def get_metrics(self):
        if self._verbose:
            print("Markov: Collecting metrics...")

        block_ratio = self._compute_blocking_ratio()
        requests_in_system = self._compute_requests_in_system()
        idling_container = self._compute_idling_container()
        effective_arrival_rate = self._compute_effective_arrival_rate(block_ratio)
        waiting_time = self._compute_waiting_time(requests_in_system, effective_arrival_rate)
        warm_hit_ratio = self._compute_warm_hit_ratio()
        cpu_usage = self._compute_resource_usage("cpu")
        ram_usage = self._compute_resource_usage("ram")
        cpu_usage_per_request = cpu_usage / effective_arrival_rate
        ram_usage_per_request = ram_usage / effective_arrival_rate
        power_usage = self._compute_power_usage(cpu_usage)
        power_usage_per_request = power_usage / effective_arrival_rate

        return {
            "blocking_ratios": [block_ratio],
            "processing_requests": [requests_in_system],
            "idling_containers": [idling_container],
            "warm_hit_ratio": [warm_hit_ratio],
            "effective_arrival_rates": [effective_arrival_rate],
            "latency": [waiting_time],
            'cpu_usage': [cpu_usage],
            'ram_usage': [ram_usage],
            'cpu_usage_per_request': [cpu_usage_per_request],
            'ram_usage_per_request': [ram_usage_per_request],
            'power_usage': [power_usage],
            'power_usage_per_request': [power_usage_per_request],
        }


if __name__ == "__main__":
    config = {
        "arrival_rate": 1,
        "service_rate": 0.5,
        "spawn_rate": 0.25,
        "theta": 0.0001,
        "max_queue": 15,  # n: total container slots
        "ram_warm": 10,
        "cpu_warm": 1,
        "ram_demand": 20,
        "cpu_demand": 25,
        "power_max": 150,
        "power_min_scale": 0.4,
    }
    m = MarkovModel(config, verbose=False)
    G = m._G
    # draw_graph_updated(G, node_size=1000)
    print(f"Total probability: {m.verify_total_probability()}")
    print(m.get_metrics())
