import simpy
import random
import numpy as np
import time
import os
import pandas as pd
from datetime import datetime

# Import simulator components
from System import System
from Topology import Topology
from Cluster import Cluster
from variables import request_stats, latency_stats

# Import Markov model
from Markov.model_3D import MarkovModel

class ModelComparison:
    def __init__(self, output_dir='comparison_results'):
        """
        Initialize the comparison framework.
        
        Args:
            output_dir (str): Directory to save comparison results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
    
    def run_scenario(self, scenario_name, config):
        """
        Run a single scenario comparing the Markov model with the simulator.
        
        Args:
            scenario_name (str): Name of the scenario for reporting
            config (dict): Configuration parameters for both models
        
        Returns:
            dict: Results of the comparison
        """
        print(f"Running scenario: {scenario_name}")
        start_time = time.time()
        
        # Extract configuration parameters
        lam = config.get('arrival_rate')
        mu = config.get('service_rate')
        alpha = config.get('spawn_rate')  
        beta = config.get('assign_rate')
        theta = config.get('timeout_rate')  
        max_queue = config.get('max_queue')
        # Parameters for the simulator
        num_servers = config.get('num_servers')
        server_cpu = config.get('server_cpu')
        server_ram = config.get('server_ram')
        sim_time = config.get('sim_time')
        cpu_demand = config.get('cpu_demand')
        ram_demand = config.get('ram_demand')
        cpu_warm = config.get('cpu_warm')
        ram_warm = config.get('ram_warm')
        random_seed = config.get('random_seed')
        
        # Create Markov model config (converting parameters as needed)
        markov_config = {
            "lam": lam,
            "mu": mu,
            "alpha": alpha,
            "beta": beta,
            "theta": theta,
            "max_queue": max_queue,
            "num_runs": 1
        }
        
        # Run Markov model
        markov_metrics = self._run_markov_model(markov_config)
        
        # Set up simulator parameters
        global REQUEST_ARRIVAL_RATE, REQUEST_SERVICE_RATE, CONTAINER_SPAWN_TIME
        global CONTAINER_IDLE_TIMEOUT, CONTAINER_ASSIGN_RATE, NUM_SERVERS
        global SERVER_CPU_CAPACITY, SERVER_RAM_CAPACITY, SIM_TIME
        global CPU_DEMAND, RAM_DEMAND, CPU_WARM, RAM_WARM, RANDOM_SEED
        
        # Save original values to restore after simulation
        # original_vals = {
        #     'REQUEST_ARRIVAL_RATE': REQUEST_ARRIVAL_RATE,
        #     'REQUEST_SERVICE_RATE': REQUEST_SERVICE_RATE,
        #     'CONTAINER_SPAWN_TIME': CONTAINER_SPAWN_TIME,
        #     'CONTAINER_IDLE_TIMEOUT': CONTAINER_IDLE_TIMEOUT,
        #     'CONTAINER_ASSIGN_RATE': CONTAINER_ASSIGN_RATE,
        #     'NUM_SERVERS': NUM_SERVERS,
        #     'SERVER_CPU_CAPACITY': SERVER_CPU_CAPACITY,
        #     'SERVER_RAM_CAPACITY': SERVER_RAM_CAPACITY,
        #     'SIM_TIME': SIM_TIME,
        #     'CPU_DEMAND': CPU_DEMAND,
        #     'RAM_DEMAND': RAM_DEMAND,
        #     'CPU_WARM': CPU_WARM,
        #     'RAM_WARM': RAM_WARM,
        #     'RANDOM_SEED': RANDOM_SEED
        # }
        
        # Set new values for this scenario
        REQUEST_ARRIVAL_RATE = lam
        REQUEST_SERVICE_RATE = mu
        CONTAINER_SPAWN_TIME = 1/alpha if alpha > 0 else 5.0
        CONTAINER_IDLE_TIMEOUT = 1/theta if theta > 0 else 2.0
        CONTAINER_ASSIGN_RATE = beta
        NUM_SERVERS = num_servers
        SERVER_CPU_CAPACITY = server_cpu
        SERVER_RAM_CAPACITY = server_ram
        SIM_TIME = sim_time
        CPU_DEMAND = cpu_demand
        RAM_DEMAND = ram_demand 
        CPU_WARM = cpu_warm
        RAM_WARM = ram_warm
        RANDOM_SEED = random_seed
        
        # Run simulator and get metrics
        simulator_metrics = self._run_simulator()
        
        # Restore original values
        # for key, val in original_vals.items():
        #     globals()[key] = val
        
        # Calculate comparison metrics
        comparison = self._compare_metrics(markov_metrics, simulator_metrics)
        
        # Add scenario info
        result = {
            'scenario_name': scenario_name,
            'config': config,
            'markov_metrics': markov_metrics,
            'simulator_metrics': simulator_metrics,
            'comparison': comparison,
            'runtime_seconds': time.time() - start_time
        }
        
        self.results.append(result)
        return result
        
    def _run_markov_model(self, config):
        """
        Run the Markov model and extract its metrics.
        
        Args:
            config (dict): Markov model configuration
            
        Returns:
            dict: Metrics from the Markov model
        """
        print("Running Markov model...")
        try:
            model = MarkovModel(config, verbose=False)
            metrics = model.get_metrics()
            # Convert lists to scalar values for easier comparison
            return {k: v[0] if isinstance(v, list) and len(v) > 0 else v 
                   for k, v in metrics.items()}
        except Exception as e:
            print(f"Error running Markov model: {e}")
            return {}
    
    def _run_simulator(self):
        """Run the simulator and extract its metrics."""
        print("Running simulator...")
        
        # Reset statistics dictionaries
        for key in request_stats:
            request_stats[key] = 0
        
        for key in latency_stats:
            latency_stats[key] = 0.0
        
        # Set up simulator environment
        random.seed(RANDOM_SEED)
        env = simpy.Environment()
        
        # For tracking waiting requests over time
        waiting_data = []
        last_time = 0
        
        # Custom monitor process to sample waiting requests
        def monitor_waiting_requests(env):
            nonlocal last_time
            while True:
                current_waiting = request_stats['generated'] - request_stats['processed'] - (
                    request_stats['blocked_no_server_capacity'] + 
                    request_stats['blocked_spawn_failed']
                )
                
                # Record the current state with time duration since last sample
                time_delta = env.now - last_time
                if last_time > 0:  # Skip the first entry
                    waiting_data.append((current_waiting, time_delta))
                
                last_time = env.now
                yield env.timeout(1.0)  # Sample every time unit
        
        # Set up topology (simplified version without topology)
        use_topology = False
        topology = None
        cluster_node = "default_node"
        
        # Create cluster and system
        cluster = Cluster(env, cluster_node, NUM_SERVERS, SERVER_CPU_CAPACITY, SERVER_RAM_CAPACITY)
        system = System(env, REQUEST_ARRIVAL_RATE, REQUEST_SERVICE_RATE, CONTAINER_SPAWN_TIME,
                       CONTAINER_IDLE_TIMEOUT, topology, cluster, use_topology=use_topology)
        
        # Start the simulation and monitoring
        env.process(system.request_generator())
        env.process(monitor_waiting_requests(env))
        env.run(until=SIM_TIME)
        
        # Calculate time-weighted average of waiting requests
        total_weight = sum(weight for _, weight in waiting_data)
        avg_waiting_requests = sum(value * weight for value, weight in waiting_data) / total_weight if total_weight > 0 else 0
        
        # Calculate other metrics
        total_blocked = request_stats['blocked_no_server_capacity'] + request_stats['blocked_spawn_failed']
        effective_arrival_rate = REQUEST_ARRIVAL_RATE * (1 - total_blocked / request_stats['generated']) if request_stats['generated'] > 0 else 0
        
        # Calculate waiting time from latency statistics
        avg_waiting_time = 0
        if latency_stats['count'] > 0:
            avg_waiting_time = latency_stats['spawning_time'] / latency_stats['count']
        
        # Construct metrics object
        metrics = {
            'blocking_ratios': total_blocked / request_stats['generated'] if request_stats['generated'] > 0 else 0,
            'waiting_requests': avg_waiting_requests,  # Use the time-weighted average
            'processing_requests': request_stats['processed'] / SIM_TIME,
            'effective_arrival_rates': effective_arrival_rate,
            'waiting_times': avg_waiting_time
        }
        
        return metrics
    
    def _compare_metrics(self, markov_metrics, simulator_metrics):
        """
        Compare metrics between Markov model and simulator.
        
        Args:
            markov_metrics (dict): Metrics from the Markov model
            simulator_metrics (dict): Metrics from the simulator
            
        Returns:
            dict: Comparison results including MSE and MAPE
        """
        comparison = {}
        
        # Ensure both metrics have the same keys
        all_keys = set(markov_metrics.keys()) | set(simulator_metrics.keys())
        
        for key in all_keys:
            if key in markov_metrics and key in simulator_metrics:
                markov_val = markov_metrics[key]
                sim_val = simulator_metrics[key]
                
                # Calculate absolute error
                abs_error = abs(markov_val - sim_val)
                
                # Calculate squared error
                squared_error = abs_error ** 2
                
                # Calculate absolute percentage error if possible
                ape = 0
                if sim_val != 0:
                    ape = (abs_error / abs(sim_val)) * 100
                
                comparison[key] = {
                    'markov_value': markov_val,
                    'simulator_value': sim_val,
                    'absolute_error': abs_error,
                    'squared_error': squared_error,
                    'absolute_percentage_error': ape
                }
        
        # Calculate MSE across all metrics
        if comparison:
            mse = sum(item['squared_error'] for item in comparison.values()) / len(comparison)
            mape = sum(item['absolute_percentage_error'] for item in comparison.values()) / len(comparison)
            
            comparison['overall'] = {
                'MSE': mse,
                'MAPE': mape
            }
        
        return comparison
    
    def run_multiple_scenarios(self, scenarios):
        """
        Run multiple scenarios and compile results.
        
        Args:
            scenarios (list): List of (scenario_name, config) tuples
            
        Returns:
            pandas.DataFrame: Summary of comparison results
        """
        results = []
        
        for scenario_name, config in scenarios:
            result = self.run_scenario(scenario_name, config)
            summary = {
                'Scenario': scenario_name,
                'MSE': result['comparison']['overall']['MSE'],
                'MAPE': result['comparison']['overall']['MAPE'],
                'Runtime(s)': result['runtime_seconds']
            }
            
            # Add individual metric comparisons
            for metric in ['blocking_ratios', 'waiting_requests', 'processing_requests', 
                          'effective_arrival_rates', 'waiting_times']:
                if metric in result['comparison']:
                    summary[f'{metric}_Markov'] = result['comparison'][metric]['markov_value']
                    summary[f'{metric}_Simulator'] = result['comparison'][metric]['simulator_value']
                    summary[f'{metric}_APE'] = result['comparison'][metric]['absolute_percentage_error']
            
            results.append(summary)
        
        # Create DataFrame from results
        df = pd.DataFrame(results)
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.output_dir, f'comparison_results_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        return df
    
    def generate_report(self, output_format='text'):
        """
        Generate a report of comparison results.
        
        Args:
            output_format (str): Format of the report ('text' or 'html')
            
        Returns:
            str: Report content
        """
        if not self.results:
            return "No results to report."
        
        if output_format == 'text':
            report = "=== Model Comparison Report ===\n\n"
            
            for result in self.results:
                report += f"Scenario: {result['scenario_name']}\n"
                report += f"Runtime: {result['runtime_seconds']:.2f} seconds\n"
                report += f"Overall MSE: {result['comparison']['overall']['MSE']:.6f}\n"
                report += f"Overall MAPE: {result['comparison']['overall']['MAPE']:.2f}%\n\n"
                
                report += "Metric Comparison:\n"
                for metric in ['blocking_ratios', 'waiting_requests', 'processing_requests', 
                              'effective_arrival_rates', 'waiting_times']:
                    if metric in result['comparison']:
                        comp = result['comparison'][metric]
                        report += f"  {metric}:\n"
                        report += f"    Markov: {comp['markov_value']:.6f}\n"
                        report += f"    Simulator: {comp['simulator_value']:.6f}\n"
                        report += f"    Absolute Error: {comp['absolute_error']:.6f}\n"
                        report += f"    APE: {comp['absolute_percentage_error']:.2f}%\n\n"
                
                report += "-" * 50 + "\n\n"
            
            return report
        
        elif output_format == 'html':
            # Implement HTML report if needed
            pass
        
        return "Unsupported output format."


if __name__ == "__main__":
    # Example usage:
    comparator = ModelComparison()
    
    # Define different scenarios to test
    scenarios = [
        # Scenario 1: Base configuration
        ("Base Configuration", {
            'arrival_rate': 10,
            'service_rate': 1,
            'spawn_rate': 1/5,
            'assign_rate': 1000,
            'timeout_rate': 0.5,
            'max_queue': 4,
            'num_servers': 2,
            'server_cpu': 100,
            'server_ram': 100,
            'sim_time': 1000,
            'cpu_demand': 49,
            'ram_demand': 49,
            'cpu_warm': 49,
            'ram_warm': 49,
            'random_seed': 42
        }),
        
        # # Scenario 2: Higher arrival rate
        # ("High Arrival Rate", {
        #     'arrival_rate': 20,
        #     'service_rate': 2,
        #     'spawn_rate': 1/5,
        #     'assign_rate': 1000,
        #     'timeout_rate': 0.5,
        #     'max_queue': 4,
        #     'num_servers': 2,
        #     'sim_time': 1000,
        #     'random_seed': 42
        # }),
        
        # # Scenario 3: Slower service rate
        # ("Slow Service Rate", {
        #     'arrival_rate': 10,
        #     'service_rate': 1,
        #     'spawn_rate': 1/5,
        #     'assign_rate': 1000, 
        #     'timeout_rate': 0.5,
        #     'max_queue': 4,
        #     'num_servers': 2,
        #     'sim_time': 1000,
        #     'random_seed': 42
        # })
    ]
    
    # Run scenarios and get summary
    summary_df = comparator.run_multiple_scenarios(scenarios)
    print(summary_df)
    
    # Generate and print report
    report = comparator.generate_report()
    print(report)
    
    # You can add more scenarios or modify the existing ones as needed
