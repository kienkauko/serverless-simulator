from Server import Server
import variables 

class Cluster:
    def __init__(self, env, name, config=None):
        self.name = name
        self.env = env           
        # self.verbose = verbose  # Flag to control logging output
        # if variables.CLUSTER_STRATEGY == "centralized_cloud":
        #     self.servers = [Server(env, self, i, config)]
        # else:            
        self.servers = [Server(env, self, i, config) for i in range(config["num_servers"])]
        self.node = config["node"]  # Topology node where this cluster is located
        self.num_servers = len(self.servers)
        self.total_cpu_capacity = self.servers[0].cpu_capacity * self.num_servers
        self.total_ram_capacity = self.servers[0].ram_capacity * self.num_servers
        # Variables for tracking resource usage
        self.update_period = 10.0  # Time units between resource usage updates. Set 0 to update at every event
        self.last_resource_update = 0.0
        self.total_cpu_usage_area = 0.0  # Time-weighted CPU usage
        self.total_ram_usage_area = 0.0  # Time-weighted RAM usage
        self.total_cpu_reserve_area = 0.0  # Time-weighted CPU reserve
        self.total_ram_reserve_area = 0.0  # Time-weighted RAM reserve
        self.total_cpu_capacity = 0.0  # Total CPU capacity of all servers
        self.total_ram_capacity = 0.0  # Total RAM capacity of all servers
        self.total_energy_usage_area = 0.0  # Time-weighted CPU usage

        self.spawn_time_factor = config.get('spawn_time_factor', 1.0)  # Default to 1.0 if not specified
        self.processing_time_factor = config.get('processing_time_factor', 1.0)  # Default to 1.0 if not specified

    def get_utilization(self):
        result = []

        used_cpu_real = sum(server.cpu_capacity - server.cpu_real for server in self.servers)
        used_ram_real = sum(server.ram_capacity - server.ram_real for server in self.servers)
        used_cpu_reserve = sum(server.cpu_capacity - server.cpu_reserve for server in self.servers)
        used_ram_reserve = sum(server.ram_capacity - server.ram_reserve for server in self.servers)
        
        result.append(used_cpu_real / self.total_cpu_capacity)
        result.append(used_ram_real / self.total_ram_capacity)
        result.append(used_cpu_reserve / self.total_cpu_capacity)
        result.append(used_ram_reserve / self.total_ram_capacity)

        return result
    
    def get_power(self):
        total_power = sum(server.get_power() for server in self.servers)
        return total_power
    
    def update_resource_stats(self):
        """Update time-weighted statistics for resource usage."""
        current_time = self.env.now
        # Either update at fixed intervals or on every event if update_period is 0
        if (current_time - self.last_resource_update) > self.update_period:
            time_delta = current_time - self.last_resource_update
            # Calculate current CPU and RAM usage across all servers
            current_cpu_usage = sum(server.cpu_capacity - server.cpu_real for server in self.servers)
            current_ram_usage = sum(server.ram_capacity - server.ram_real for server in self.servers)
            current_cpu_reserve = sum(server.cpu_capacity - server.cpu_reserve for server in self.servers)
            current_ram_reserve = sum(server.ram_capacity - server.ram_reserve for server in self.servers)
            current_power_usage = sum(server.get_power() for server in self.servers)
            # Update time-weighted usage areas
            self.total_cpu_usage_area += time_delta * current_cpu_usage
            self.total_ram_usage_area += time_delta * current_ram_usage
            self.total_cpu_reserve_area += time_delta * current_cpu_reserve
            self.total_ram_reserve_area += time_delta * current_ram_reserve
            self.total_energy_usage_area += time_delta * current_power_usage
            self.last_resource_update = current_time

    def get_mean_cpu(self, result_type, resource_type='usage'):
        """Calculate the mean CPU usage over time."""
        # Update statistics first to include latest data
        # self.update_resource_stats()
        
        if result_type == 'cluster':
            if resource_type == 'usage':
                return self.total_cpu_usage_area / (self.env.now)
            else:
                return self.total_cpu_reserve_area / (self.env.now)
        else:
            if resource_type == 'usage':
                return self.total_cpu_usage_area / (self.env.now*self.total_cpu_capacity)
            else:
                return self.total_cpu_reserve_area / (self.env.now*self.total_cpu_capacity)
        
    def get_mean_ram(self, result_type, resource_type='usage'):
        """Calculate the mean RAM usage over time."""
        # Update statistics first to include latest data
        # self.update_resource_stats()
        
        if result_type == 'cluster':
            if resource_type == 'usage':
                return self.total_ram_usage_area / (self.env.now)
            else:
                return self.total_ram_reserve_area / (self.env.now)
        else:
            if resource_type == 'usage':
                return self.total_ram_usage_area / (self.env.now*self.total_ram_capacity)
            else:
                return self.total_ram_reserve_area / (self.env.now*self.total_ram_capacity)
    

    def get_mean_power(self, type='cluster'):
        """Calculate the mean CPU usage over time."""
        # Update statistics first to include latest data
        # self.update_resource_stats()
        # if variables.CLUSTER_STRATEGY == "centralized_cloud":
        #     print("WARNING: Current strategy is centralized cloud, power is not correct!")
        if type == 'cluster':
            return self.total_energy_usage_area / (self.env.now)
        else:
            # on_servers = sum(1 for server in self.servers if server.is_ON())
            return self.total_energy_usage_area / (self.env.now*len(self.servers))