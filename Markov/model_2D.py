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
from scipy.optimize import brentq
import math
from sympy import symbols, Matrix, simplify, factor, cancel, solve, Eq, pprint, latex, factorial
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
        self._spawn_distribution = config.get("spawn_distribution", "exponential")
        self._max_queue_warm = config["queue_warm"]
        self._max_queue_cold = config["queue_cold"]
        self._ram_warm = config["ram_warm"]
        self._cpu_warm = config["cpu_warm"]
        self._ram_active = config["ram_demand"]
        self._cpu_active = config["cpu_demand"]
        self._peak_power = config["peak_power"]
        self._power_scale = config["power_scale"]
        self._cpu_transit = config["cpu_transit"]
        self._ram_transit = config["ram_transit"]
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
        string_lambda = r'$\lambda$'
        string_mu = r"$\mu$"
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
    
    def compute_symbolic_steady_state(self):
        """
        Build the rate matrix symbolically using SymPy and solve the global
        balance equations  pi @ Q = 0,  sum(pi) = 1  to obtain closed-form
        expressions for every steady-state probability.

        Returns:
            dict: mapping  (i, j) state tuple -> simplified SymPy expression
        """
        # --- 1. Symbolic parameters ---
        lam, mu, alpha = symbols('lambda mu alpha', positive=True, real=True)

        # --- 2. Ordered state list (same order used by create_rate_matrix) ---
        nodes = sorted(list(self._G.nodes))
        num_states = len(nodes)
        n2i = {node: idx for idx, node in enumerate(nodes)}

        # --- 3. Build symbolic rate matrix Q ---
        Q_sym = Matrix.zeros(num_states, num_states)

        for edge in self._G.edges:
            src, dst = edge
            i0 = n2i[src]
            i1 = n2i[dst]

            # Reconstruct the symbolic rate from the transition type
            si, sj = src
            di, dj = dst

            if di == si + 1 and dj == sj:
                # Arrival: rate = lambda
                rate = lam
            elif di == si and dj == sj + 1:
                # Arrival overflows to cold queue: rate = lambda
                rate = lam
            elif di == si and dj == sj - 1:
                # Cold completion: rate = j * alpha
                rate = sj * alpha
            elif di == si - 1 and dj == sj:
                # Warm completion: rate = i * mu
                rate = si * mu
            else:
                # Fallback (should not happen for this model)
                rate = self._G[src][dst]["weight"]

            Q_sym[i0, i1] = rate

        # Fill diagonal so each row sums to zero
        for r in range(num_states):
            Q_sym[r, r] = -sum(Q_sym[r, c] for c in range(num_states) if c != r)

        # --- 4. Symbolic probability variables ---
        pi_syms = symbols(f'pi0:{num_states}', positive=True, real=True)

        # --- 5. Balance equations:  pi @ Q = 0 ---
        pi_vec = Matrix([list(pi_syms)])          # 1 x n row vector
        balance_vec = pi_vec * Q_sym               # 1 x n
        equations = [Eq(balance_vec[0, col], 0) for col in range(num_states)]

        # Normalization
        equations.append(Eq(sum(pi_syms), 1))

        # --- 6. Solve ---
        solution = solve(equations, pi_syms, dict=True)

        if not solution:
            print("SymPy could not find a symbolic solution.")
            return {}

        sol = solution[0]  # first (and typically only) solution dict

        # --- 7. Map back to state tuples, simplify, and express compactly ---
        #
        # For the birth-death structure of this model, the raw SymPy output
        # expands everything into large integer coefficients (factorials).
        # We recognise the closed-form pattern directly:
        #
        #   For a pure cold chain (n_w = 0):
        #       π(0,j) = (λ/α)^j / j!  /  Σ_{k=0}^{n_c} (λ/α)^k / k!
        #
        #   For the general 2-D chain the same idea applies per dimension,
        #   but SymPy's generic solve already gives the exact answer — we
        #   just need better simplification.
        #
        # Strategy: try factor → cancel → simplify, keep the shortest repr.

        def _best_simplify(expr):
            """Return the most compact form among several simplification strategies."""
            candidates = [
                factor(expr),
                cancel(expr),
                simplify(expr),
            ]
            # Pick the representation with the shortest string length
            return min(candidates, key=lambda e: len(str(e)))

        result = {}
        for node in nodes:
            idx = n2i[node]
            result[node] = _best_simplify(sol[pi_syms[idx]])

        # --- 7b. Build factorial-based compact form ---
        # For states reachable purely along one dimension the closed form is
        #   π(i,j) = ρ_w^i / i! · ρ_c^j / j!  ·  π(0,0)
        # where ρ_w = λ/μ, ρ_c = λ/α.  Express this alongside the raw result.
        rho_w = lam / mu
        rho_c = lam / alpha
        pi00_sym = result.get((0, 0), None)
        compact = {}
        if pi00_sym is not None:
            for node in nodes:
                i, j = node
                compact[node] = (rho_w**i / factorial(i)) * (rho_c**j / factorial(j)) * pi00_sym

        # --- 8. Pretty-print ---
        print("\n=== Closed-Form Steady-State Probabilities ===")
        print(f"    Symbols:  λ = arrival rate,  μ = warm service rate,  "
              f"α = cold completion rate")
        print(f"    ρ_w = λ/μ,  ρ_c = λ/α")
        print(f"    State space:  (warm_jobs, cold_jobs)\n")

        for node in nodes:
            print(f"  π{node} =")
            pprint(result[node], use_unicode=True)
            print(f"        LaTeX:  {latex(result[node])}")
            if node in compact:
                i, j = node
                print(f"        Compact:  (ρ_w^{i}/{i}!) · (ρ_c^{j}/{j}!) · π(0,0)")
            print()

        if pi00_sym is not None:
            print(f"  Where π(0,0) =")
            pprint(pi00_sym, use_unicode=True)
            print(f"        LaTeX:  {latex(pi00_sym)}\n")

        print("================================================\n")

        return result

    def validate_symbolic_vs_numeric(self):
        """
        Validate symbolic closed-form probabilities against the numerically
        computed ones.  Substitutes the actual (λ, μ, α) values into each
        symbolic expression and prints the results side by side.
        """
        symbolic = self.compute_symbolic_steady_state()
        if not symbolic:
            print("No symbolic solution to validate.")
            return

        lam_sym, mu_sym, alpha_sym = symbols('lambda mu alpha',
                                              positive=True, real=True)
        subs_dict = {
            lam_sym:   self._lam,
            mu_sym:    self._mu,
            alpha_sym: self._alpha,
        }

        nodes = sorted(symbolic.keys())
        print("=== Validation: Symbolic vs Numeric ===")
        print(f"    λ = {self._lam},  μ = {self._mu},  α = {self._alpha:.6f}\n")

        max_err = 0.0
        for node in nodes:
            analytic = float(symbolic[node].subs(subs_dict))
            numeric  = float(self._state_probabilities[node])
            err = abs(analytic - numeric)
            max_err = max(max_err, err)
            match = "✓" if err < 1e-10 else "✗"
            print(f"  π{node}:  analytical = {analytic:.6f},  "
                  f"numeric = {numeric:.6f}  {match}")

        print(f"\n  Max absolute error: {max_err:.2e}")
        print("========================================\n")

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
        # print("BLOCKING STATE", s)
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
            transit = self._cpu_transit
        elif resource == "ram":
            active = self._ram_active
            warm = self._ram_warm
            transit = self._ram_transit
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

    def _compute_p_warm_cold(self):
        """
        Compute probabilities that an admitted (non-blocked) request
        is served by a warm function (p_w) or cold-started (p_c).
        
        p_w = P(warm) / (1 - p_b)
        p_c = P(cold) / (1 - p_b)
        """
        p_b = self._compute_blocking_ratio()
        nw = self._max_queue_warm
        nc = self._max_queue_cold
        
        # P(warm): arrival finds warm pool not full
        P_warm = 0.0
        for i in range(nw):  # i = 0 to n_w - 1
            for j in range(nc + 1):  # j = 0 to n_c
                P_warm += self._state_probabilities[i, j]
        
        # P(cold): warm pool full but cold pool not full
        P_cold = 0.0
        for j in range(nc):  # j = 0 to n_c - 1
            P_cold += self._state_probabilities[nw, j]
        
        if p_b < 1.0:
            p_w = P_warm / (1.0 - p_b)
            p_c = P_cold / (1.0 - p_b)
        else:
            p_w = 0.0
            p_c = 0.0
        
        return p_w, p_c

    def _compute_variance_latency(self):
        """
        Compute Var[B_r] using the law of total variance (mixture approach).
        
        B_r is a mixture:
          - B_w           with prob p_w (warm-served)
          - B_c + B_w     with prob p_c (cold-started)
        
        Var[B_r] = p_w * Var[B_w] + p_c * Var[B_c + B_w] + p_w * p_c * (E[B_c])^2
        
        Depends on spawn_distribution:
          - "exponential" (M): Var[B_c] = (E[B_c])^2
            => Var[B_r] = 1/mu^2 + p_c*(2 - p_c)*(E[B_c])^2
          - "deterministic" (D): Var[B_c] = 0
            => Var[B_r] = 1/mu^2 + p_w*p_c*(E[B_c])^2
        """
        p_w, p_c = self._compute_p_warm_cold()
        mu = self._mu
        E_Bc = 1.0 / self._spawn_rate
        
        Var_Bw = 1.0 / (mu ** 2)  # Var of Exp(mu)
        
        if self._spawn_distribution == "deterministic":
            # D case: Var[B_c] = 0, so Var[B_c + B_w] = Var[B_w]
            Var_Bc_Bw = Var_Bw
        else:
            # M case: Var[B_c] = E[B_c]^2, so Var[B_c + B_w] = E[B_c]^2 + Var[B_w]
            Var_Bc_Bw = E_Bc ** 2 + Var_Bw
        
        # Law of total variance:
        # within-component + between-component
        variance = p_w * Var_Bw + p_c * Var_Bc_Bw + p_w * p_c * (E_Bc ** 2)
        
        return variance

    def _response_time_cdf(self, t):
        """
        CDF of response time for served (non-blocked) requests.
        
        F_{B_r}(t) = p_w * F_{B_w}(t) + p_c * F_{B_c + B_w}(t)
        
        Where F_{B_c + B_w}(t) depends on spawn_distribution:
          - "exponential": hypoexponential CDF
          - "deterministic": shifted exponential CDF
        """
        p_w, p_c = self._compute_p_warm_cold()
        mu = self._mu
        mu_c = self._spawn_rate  # cold-start rate
        
        # Warm path: exponential CDF
        F_warm = 1.0 - np.exp(-mu * t)
        
        # Cold path CDF
        if self._spawn_distribution == "deterministic":
            d = 1.0 / mu_c  # deterministic cold-start duration
            if t < d:
                F_cold = 0.0
            else:
                F_cold = 1.0 - np.exp(-mu * (t - d))
        else:
            # Hypoexponential: sum of two independent exponentials
            if abs(mu - mu_c) > 1e-10:
                F_cold = (1.0 
                          - (mu / (mu - mu_c)) * np.exp(-mu_c * t) 
                          + (mu_c / (mu - mu_c)) * np.exp(-mu * t))
            else:
                # Special case mu == mu_c: Erlang-2
                F_cold = 1.0 - (1.0 + mu * t) * np.exp(-mu * t)
        
        return p_w * F_warm + p_c * F_cold

    def _compute_tail_latency(self, p=0.99):
        """
        Compute the p-th percentile of response time by solving F_{B_r}(t_p) = p.
        
        Args:
            p (float): Percentile, e.g. 0.95 for P95, 0.99 for P99.
        
        Returns:
            float: t_p such that P(B_r <= t_p) = p
        """
        mu = self._mu
        mu_c = self._spawn_rate
        # Upper bound for root search
        t_max = 50.0 * (1.0 / mu + 1.0 / mu_c)
        
        try:
            t_p = brentq(lambda t: self._response_time_cdf(t) - p, 1e-10, t_max)
        except ValueError:
            t_p = float('inf')
        
        return t_p

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
        variance_latency = self._compute_variance_latency()
        std_latency = np.sqrt(max(variance_latency, 0))
        p_w, p_c = self._compute_p_warm_cold()
        p50 = self._compute_tail_latency(0.50)
        p95 = self._compute_tail_latency(0.95)
        p99 = self._compute_tail_latency(0.99)
        # cpu_usage = self._compute_cpu_usage()
        # ram_usage = self._compute_ram_usage()
        cpu_usage_per_request = cpu_usage/requests_in_system
        ram_usage_per_request = ram_usage/requests_in_system

        return {
            "blocking_ratios": [block_ratio],
            "requests_in_system": [requests_in_system],
            "latency": [latency],
            "variance_latency": [variance_latency],
            "std_latency": [std_latency],
            "p_warm": [p_w],
            "p_cold": [p_c],
            "p50_latency": [p50],
            "p95_latency": [p95],
            "p99_latency": [p99],
            "cpu_usage": [cpu_usage],
            "ram_usage": [ram_usage],
            "power_usage": [power_usage],
            # "processing_requests": [processing_requests],
            # "idling_requests": [idling_requests],
            # "effective_arrival_rates": [effective_arrival_rate],
            # "waiting_times": [waiting_time],
            # 'mean_cpu_usage': [cpu_usage],
            # 'mean_ram_usage': [ram_usage],
            'mean_cpu_usage_per_request': [cpu_usage_per_request],
            'mean_ram_usage_per_request': [ram_usage_per_request],
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
        "lam": 0.5,
        "mu": 1/2.50,
        "spawn_rate": 1/12.0, #default: 1/9.85
        "queue_warm":2, # queue
        "queue_cold":2, # queue
        "serving_time": "exponential",
        "spawn_distribution": "deterministic",  # "exponential" or "deterministic"
        "arrivals": "exponential",
        # "lam_factor": 1,
        "ram_warm": 2.60,
        "cpu_warm": 0.01,
        "ram_demand": 2.59,
        "cpu_demand": 5.0,
        "cpu_transit": 5.35,
        "ram_transit": 1.51,
        "peak_power": 150.0,
        "power_scale": 0.2,  # Power scale factor

    }
    m = MarkovModel(config, verbose=False)
    G = m._G
    #draw_graph_updated(G, node_size=1000)
    metrics = m.get_metrics()
    print("\n=== Markov Model Metrics ===")
    print(f"  Blocking ratio:        {metrics['blocking_ratios'][0]:.4f}")
    print(f"  Requests in system:    {metrics['requests_in_system'][0]:.4f}")
    print(f"  Mean latency:          {metrics['latency'][0]:.4f} s")
    print(f"  Latency std dev:       {metrics['std_latency'][0]:.4f} s")
    print(f"  P50 latency:           {metrics['p50_latency'][0]:.4f} s")
    print(f"  P95 latency:           {metrics['p95_latency'][0]:.4f} s")
    print(f"  P99 latency:           {metrics['p99_latency'][0]:.4f} s")
    print(f"  P(warm served):        {metrics['p_warm'][0]:.4f}")
    print(f"  P(cold started):       {metrics['p_cold'][0]:.4f}")
    print(f"  CPU usage:             {metrics['cpu_usage'][0]:.4f} %")
    print(f"  RAM usage:             {metrics['ram_usage'][0]:.4f} %")
    print(f"  Power usage:           {metrics['power_usage'][0]:.4f} W")
    print("==========================\n")

    # Validate symbolic closed-form vs numeric steady-state probabilities
    m.validate_symbolic_vs_numeric()
