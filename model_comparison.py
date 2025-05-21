import simpy
import random
import numpy as np
import time
import os
import pandas as pd
from datetime import datetime
import math

# Import simulator components
from System import System
from Server import Server
from variables import request_stats, latency_stats, config

# Import Markov model
from Markov.model_3D import MarkovModel

class ModelComparison:
    def __init__(self, output_dir='comparison_results', verbose=False):
        """
        Initialize the comparison framework.
        
        Args:
            output_dir (str): Directory to save comparison results
        """
        self.output_dir = output_dir
        self.verbose = verbose
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
    
    def run_scenario(self, scenario_name, config_params):
        """
        Run a single scenario comparing the Markov model with the simulator.
        
        Args:
            scenario_name (str): Name of the scenario for reporting
            config_params (dict): Configuration parameters for both models
        
        Returns:
            dict: Results of the comparison
        """
        print(f"Running scenario: {scenario_name}")
        start_time = time.time()
        
        # Extract configuration parameters
        lam = config_params.get('arrival_rate')
        mu = config_params.get('service_rate')
        alpha = config_params.get('spawn_rate')  
        beta = config_params.get('arrival_rate') # assign rate equals arrival rate
        theta = config_params.get('timeout_rate')  
        # Parameters for the simulator
        num_servers = config_params.get('num_servers')
        server_cpu = config_params.get('server_cpu')
        server_ram = config_params.get('server_ram')
        sim_time = config_params.get('sim_time')
        cpu_demand = config_params.get('cpu_demand')
        ram_demand = config_params.get('ram_demand')
        cpu_warm = config_params.get('cpu_warm')
        ram_warm = config_params.get('ram_warm')
        cpu_transit = config_params.get('cpu_transit')
        ram_transit = config_params.get('ram_transit')
        distribution = config_params.get('distribution')
        random_seed = config_params.get('random_seed')
        
        # Calculate max_queue dynamically based on the formula
        max_queue = num_servers * math.floor(100/max(cpu_demand, ram_demand))
        # print(f"Max queue size: {max_queue} for {num_servers} servers")
        # print(f"CPU demand: {cpu_demand}, RAM demand: {ram_demand}")
        # Create Markov model config (converting parameters as needed)
        markov_config = {
            "lam": lam,
            "mu": mu,
            "alpha": alpha,
            "beta": beta,
            "theta": theta,
            "max_queue": max_queue,
            "ram_warm": ram_warm,
            "cpu_warm": cpu_warm,
            "ram_demand": ram_demand,
            "cpu_demand": cpu_demand,
            "num_runs": 1
        }
        print(f"Markov model config: {markov_config}")
        # Run Markov model
        markov_metrics = self._run_markov_model(markov_config)
        
        # Create a simulator configuration based on the passed parameters
        simulator_config = {
            "system": {
                "num_servers": num_servers,
                "sim_time": sim_time,
                "distribution": distribution,
                "verbose": self.verbose,
            },
            "server": {
                "cpu_capacity": server_cpu,
                "ram_capacity": server_ram,
            },
            "request": {
                "arrival_rate": lam,
                "service_rate": mu,
                "bandwidth_demand": 10.0,  # Default value
                "cpu_warm": cpu_warm,
                "ram_warm": ram_warm,
                "cpu_demand": cpu_demand,
                "ram_demand": ram_demand,
                "cpu_transit": cpu_transit,
                "ram_transit": ram_transit,
            },
            "container": {
                "spawn_time": 1/alpha if alpha > 0 else exit(1),
                "idle_timeout": 1/theta if theta > 0 else exit(1),
            },
            "topology": {
                "use_topology": False,
            }
        }
        
        # Run simulator with the config
        simulator_metrics = self._run_simulator(simulator_config, random_seed)
        
        # Calculate comparison metrics
        comparison = self._compare_metrics(markov_metrics, simulator_metrics)
        
        # Add scenario info
        result = {
            'scenario_name': scenario_name,
            'config': config_params,
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
    
    def _run_simulator(self, sim_config, random_seed, num_repetitions=10):
        """
        Run the simulator multiple times and extract averaged metrics.
        
        Args:
            sim_config (dict): Simulator configuration
            random_seed (int): Base random seed for reproducibility
            num_repetitions (int): Number of repetitions to run the simulator
            
        Returns:
            dict: Averaged metrics from the simulator runs
        """
        print(f"Running simulator with {num_repetitions} repetitions...")
        
        # Initialize container for storing metrics from each run
        all_metrics = {
            'blocking_ratios': [],
            'waiting_requests': [],
            'effective_arrival_rates': [],
            'waiting_times': [],
            'mean_cpu_usage': [],
            'mean_ram_usage': []
        }
        
        # Run the simulator for the specified number of repetitions
        for rep in range(num_repetitions):
            # Use a different seed for each repetition but derived from the base seed
            current_seed = random_seed + rep
            
            # Reset statistics dictionaries
            for key in request_stats:
                request_stats[key] = 0
            
            for key in latency_stats:
                latency_stats[key] = 0.0
            
            # Set up simulator environment
            random.seed(current_seed)
            env = simpy.Environment()
            
            # Create system with the config
            system = System(env, sim_config, sim_config["system"]["distribution"], sim_config["system"]["verbose"])
            
            # Add servers directly to the system
            if rep == 0:  # Only print configuration details on the first run
                print("Request arrival rate:", sim_config["request"]["arrival_rate"])
                print("Request service rate:", sim_config["request"]["service_rate"])
                print("Container spawn time:", sim_config["container"]["spawn_time"])
                print("Container idle timeout:", sim_config["container"]["idle_timeout"])
                print("Number of servers:", sim_config["system"]["num_servers"])
                print("Server CPU capacity:", sim_config["server"]["cpu_capacity"])
                print("Server RAM capacity:", sim_config["server"]["ram_capacity"])
                print("Simulation time:", sim_config["system"]["sim_time"])
                print("CPU demand:", sim_config["request"]["cpu_demand"])
                print("RAM demand:", sim_config["request"]["ram_demand"])
                print("CPU warm:", sim_config["request"]["cpu_warm"])
                print("RAM warm:", sim_config["request"]["ram_warm"])
                print("Distribution:", sim_config["system"]["distribution"])
            
            for i in range(sim_config["system"]["num_servers"]):
                server = Server(env, i, sim_config["server"]["cpu_capacity"], sim_config["server"]["ram_capacity"])
                system.add_server(server)
            
            # Start the simulation
            env.process(system.request_generator())
            env.run(until=sim_config["system"]["sim_time"])
            
            # Use the system's built-in waiting requests tracking
            # Make sure system updates its statistics one final time
            system.update_waiting_stats()
            avg_waiting_requests = system.total_waiting_area / env.now
            
            # Calculate other metrics
            total_blocked = request_stats['blocked_no_server_capacity']
            effective_arrival_rate = sim_config["request"]["arrival_rate"] * (1 - total_blocked / request_stats['generated']) if request_stats['generated'] > 0 else 0
            
            # Calculate waiting time from latency statistics
            avg_waiting_time = 0
            if latency_stats['count'] > 0:
                avg_waiting_time = latency_stats['waiting_time'] / latency_stats['count']
            
            # Calculate mean CPU and RAM usage using the new methods in the System class
            system.update_resource_stats()  # Ensure latest resource stats are collected
            mean_cpu_usage = system.get_mean_cpu_usage()
            mean_ram_usage = system.get_mean_ram_usage()
            
            # Collect metrics from this run
            all_metrics['blocking_ratios'].append(total_blocked / request_stats['generated'] if request_stats['generated'] > 0 else 0)
            all_metrics['waiting_requests'].append(avg_waiting_requests)
            all_metrics['effective_arrival_rates'].append(effective_arrival_rate)
            all_metrics['waiting_times'].append(avg_waiting_time)
            all_metrics['mean_cpu_usage'].append(mean_cpu_usage)
            all_metrics['mean_ram_usage'].append(mean_ram_usage)
            
            print(f"Completed run {rep+1}/{num_repetitions}")
        
        # Calculate average metrics across all runs
        metrics = {
            'blocking_ratios': np.mean(all_metrics['blocking_ratios']),
            'waiting_requests': np.mean(all_metrics['waiting_requests']),
            'effective_arrival_rates': np.mean(all_metrics['effective_arrival_rates']),
            'waiting_times': np.mean(all_metrics['waiting_times']),
            'mean_cpu_usage': np.mean(all_metrics['mean_cpu_usage']),
            'mean_ram_usage': np.mean(all_metrics['mean_ram_usage'])
        }
        
        # Also calculate standard deviations for reference
        metrics_std = {
            'blocking_ratios_std': np.std(all_metrics['blocking_ratios']),
            'waiting_requests_std': np.std(all_metrics['waiting_requests']),
            'effective_arrival_rates_std': np.std(all_metrics['effective_arrival_rates']),
            'waiting_times_std': np.std(all_metrics['waiting_times']),
            'mean_cpu_usage_std': np.std(all_metrics['mean_cpu_usage']),
            'mean_ram_usage_std': np.std(all_metrics['mean_ram_usage'])
        }
        
        print("Average simulator metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f} (std: {metrics_std[key+'_std']:.6f})")
        
        return metrics
    
    def _compare_metrics(self, markov_metrics, simulator_metrics):
        """
        Compare metrics between Markov model and simulator.
        
        Args:
            markov_metrics (dict): Metrics from the Markov model
            simulator_metrics (dict): Metrics from the simulator (averaged from multiple runs)
            
        Returns:
            dict: Comparison results including MSE and MAPE
        """
        comparison = {}
        squared_errors = []
        percentage_errors = []
        
        # Ensure both metrics have the same keys
        all_keys = set(markov_metrics.keys()) | set(simulator_metrics.keys())
        
        for key in all_keys:
            if key in markov_metrics and key in simulator_metrics:
                markov_val = markov_metrics[key]
                sim_val = simulator_metrics[key]
                
                # Calculate absolute error
                abs_error = abs(markov_val - sim_val)
                
                # Calculate squared error
                squared_error = (markov_val - sim_val) ** 2
                squared_errors.append(squared_error)
                
                # Calculate absolute percentage error if possible
                ape = 0
                if sim_val != 0:
                    ape = (abs_error / abs(sim_val)) * 100
                    percentage_errors.append(ape)
                
                comparison[key] = {
                    'markov_value': markov_val,
                    'simulator_value': sim_val,
                    'absolute_error': abs_error,
                    'squared_error': squared_error,
                    'absolute_percentage_error': ape
                }
        
        # Calculate MSE across all metrics
        if comparison:
            # Mean Squared Error
            mse = sum(squared_errors) / len(squared_errors) if squared_errors else 0
            # Mean Absolute Percentage Error
            mape = sum(percentage_errors) / len(percentage_errors) if percentage_errors else 0
            
            comparison['overall'] = {
                'MSE': mse,
                'MAPE': mape,
                'RMSE': np.sqrt(mse)  # Root Mean Squared Error
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
            for metric in ['blocking_ratios', 'waiting_requests', 
                          'effective_arrival_rates', 'waiting_times',
                          'mean_cpu_usage', 'mean_ram_usage']:
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
        
        # Calculate and display per-metric MAPE
        self.calculate_per_metric_mape(df)
        
        return df
    
    def calculate_per_metric_mape(self, results_df):
        """
        Calculate the Mean Absolute Percentage Error (MAPE) for each metric across all scenarios.
        
        Args:
            results_df (pandas.DataFrame): DataFrame containing comparison results
            
        Returns:
            dict: Dictionary of metrics and their average MAPE values
        """
        metrics = ['blocking_ratios', 'waiting_requests', 'effective_arrival_rates', 
                   'waiting_times', 'mean_cpu_usage', 'mean_ram_usage']
        
        per_metric_mape = {}
        print("\n=== Per-Metric MAPE Across All Scenarios ===")
        
        for metric in metrics:
            ape_column = f'{metric}_APE'
            if ape_column in results_df.columns:
                avg_mape = results_df[ape_column].mean()
                per_metric_mape[metric] = avg_mape
                print(f"{metric}: {avg_mape:.2f}%")
        
        # Sort metrics by MAPE value for clearer presentation
        sorted_metrics = sorted(per_metric_mape.items(), key=lambda x: x[1], reverse=True)
        print("\n=== Metrics Ranked by MAPE (Highest to Lowest) ===")
        for metric, mape in sorted_metrics:
            print(f"{metric}: {mape:.2f}%")
            
        return per_metric_mape
    
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
                for metric in ['blocking_ratios', 'waiting_requests', 
                              'effective_arrival_rates', 'waiting_times',
                              'mean_cpu_usage', 'mean_ram_usage']:
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
    
    # Define parameter ranges for random scenario generation
    arrival_rate_range = [1, 20]          # Requests per second
    service_rate_range = [0.5, 20]         # Service rate (requests per second)
    spawn_rate_range = [1/20, 1/1]        # Container spawn rate (containers per second)
    timeout_rate_range = [1/30, 1/2]         # Container timeout rate (timeouts per second)
    num_servers_range = [1, 50]            # Number of servers
    # server_cpu_range = [50, 200]          # CPU capacity
    # server_ram_range = [50, 200]          # RAM capacity
    # sim_time_range = [500, 1500]          # Simulation time
    cpu_demand_range = [10, 80]           # CPU demand for cold containers
    ram_demand_range = [10, 80]           # RAM demand for cold containers
    # distribution_types = ['exponential'] # Distribution types
    
    # Define both fixed and random scenarios
    scenarios = []
    
    # Generate 10 random scenarios
    random.seed(123)  # Fixed seed for reproducibility
    for i in range(1):
        # Randomly select parameters from defined ranges
        arrival_rate = round(random.uniform(*arrival_rate_range), 3)
        service_rate = round(random.uniform(*service_rate_range), 3)
        spawn_rate = round(random.uniform(*spawn_rate_range), 3)
        timeout_rate = round(random.uniform(*timeout_rate_range), 3)
        num_servers = random.randint(*num_servers_range)
        # server_cpu = random.randint(*server_cpu_range)
        # server_ram = random.randint(*server_ram_range)
        server_cpu = 100  # Fixed CPU for simplicity
        server_ram = 100
        sim_time = 500  # Fixed simulation time for simplicity
        
        # Ensure warm resources are less than demand resources
        cpu_demand = random.randint(*cpu_demand_range)
        ram_demand = random.randint(*ram_demand_range)
        
        # Warm resources are a percentage (50-90%) of demand resources
        warm_percentage = random.uniform(0.1, 0.9)
        cpu_warm = int(cpu_demand * warm_percentage)
        ram_warm = int(ram_demand * warm_percentage)
        transit_percentage = random.uniform(0.1, 0.9)
        cpu_transit = int(cpu_demand * transit_percentage)
        ram_transit = int(ram_demand * transit_percentage)
        distribution = 'deterministic'  # Fixed distribution for simplicity
        
        # Create random scenario
        random_scenario = (
            f"Random Scenario {i+1}",
            {
                'arrival_rate': arrival_rate,
                'service_rate': service_rate,
                'spawn_rate': spawn_rate,
                'timeout_rate': timeout_rate,
                'num_servers': num_servers,
                'server_cpu': server_cpu,
                'server_ram': server_ram,
                'sim_time': sim_time,
                'cpu_demand': cpu_demand,
                'ram_demand': ram_demand,
                'cpu_warm': cpu_warm,
                'ram_warm': ram_warm,
                'cpu_transit': cpu_transit,
                'ram_transit': ram_transit,
                'random_seed': 42 + i,  # Different seed for each scenario
                'distribution': distribution
            }
        )
        
        scenarios.append(random_scenario)
    
    # Run scenarios and get summary
    summary_df = comparator.run_multiple_scenarios(scenarios)
    print(summary_df)
    
    # Calculate aggregate error metrics across all random scenarios
    random_scenarios_df = summary_df[summary_df['Scenario'].str.contains('Random')]
    if not random_scenarios_df.empty:
        avg_mse = random_scenarios_df['MSE'].mean()
        avg_mape = random_scenarios_df['MAPE'].mean()
        print(f"\nAggregate Error Metrics for Random Scenarios:")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average MAPE: {avg_mape:.2f}%")
    
    # Generate and print report
    report = comparator.generate_report()
    print(report)
