import abc
import random
from typing import Optional, List, Dict, Any
from variables import request_stats, app_stats

class Scheduler(abc.ABC):
    """
    Abstract base class for container scheduling strategies.
    
    This class defines the interface for different scheduling algorithms
    that determine where to place containers and how to manage their lifecycle.
    """
    
    def __init__(self, env, cluster):
        """
        Initialize the scheduler.
        
        Args:
            env: SimPy environment
            cluster: Cluster instance containing servers
        """
        self.env = env
        self.cluster = cluster
        
    @abc.abstractmethod
    def find_server_for_spawn(self, request):
        """
        Find a suitable server to spawn a new container for the request.
        
        Args:
            request: The request that needs a container
            
        Returns:
            Server instance or None if no suitable server found
        """
        pass
    
    @abc.abstractmethod
    def calculate_idle_timeout(self, container):
        """
        Calculate the idle timeout for a container.
        
        Args:
            container: The container that has just become idle
            
        Returns:
            float: The timeout value in simulation time units
        """
        pass
    
    def spawn_container_for_request(self, request, system, path=None):
        """
        Attempt to spawn a new container for a request when no idle containers are available.
        
        Args:
            request: The request that needs a container
            system: Reference to the system to initiate container spawning
            path: Optional network path for topology-aware scheduling
            
        Returns:
            Server instance or None if no suitable server was found
        """
        # No idle container; spawn a new one using a server from the cluster.
        print(f"{self.env.now:.2f} - No idle container found for {request}. Attempting to spawn.")
        
        # Use the scheduler to find a server
        server = self.find_server_for_spawn(request)
        
        if server:
            print(f"{self.env.now:.2f} - Found potential {server} for spawning container for {request}")
            request_stats['container_spawns_initiated'] += 1
            app_stats[request.app_id]['container_spawns_initiated'] += 1
            
            # Start the spawn process (now using the server's method)
            self.env.process(server.spawn_container_process(system, request, path))
            return True
        else:
            return False
    
    def get_stats(self):
        """
        Get statistics about the scheduler's performance.
        
        Returns:
            dict: Performance metrics
        """
        return {}


class FirstFitScheduler(Scheduler):
    """
    A simple scheduler that places containers on the first server with enough capacity.
    
    This implementation follows a First-Fit strategy and uses exponentially distributed
    idle timeouts with a configurable mean value.
    """
    
    def __init__(self, env, cluster, idle_timeout_mean=2.0):
        """
        Initialize the scheduler.
        
        Args:
            env: SimPy environment
            cluster: Cluster instance containing servers
            idle_timeout_mean: Mean value for exponentially distributed idle timeouts
        """
        super().__init__(env, cluster)
        self.idle_timeout_mean = idle_timeout_mean
        self._stats = {
            'placement_attempts': 0,
            'placement_successes': 0,
            'placement_failures': 0,
            'timeouts_set': 0
        }
    
    def find_server_for_spawn(self, request):
        """
        Find the first server with enough current capacity (First-Fit).
        
        Args:
            request: The request that needs a container
            
        Returns:
            Server instance or None if no suitable server is found
        """
        self._stats['placement_attempts'] += 1
        
        for server in self.cluster.servers:
            if server.has_capacity(request.cpu_demand, request.ram_demand, 
                                   request.cpu_warm, request.ram_warm):
                self._stats['placement_successes'] += 1
                return server
        
        self._stats['placement_failures'] += 1
        return None
    
    def calculate_idle_timeout(self, container):
        """
        Calculate an exponentially distributed idle timeout.
        
        Args:
            container: The container that has just become idle
            
        Returns:
            float: The timeout value in simulation time units
        """
        self._stats['timeouts_set'] += 1
        return random.expovariate(1.0/self.idle_timeout_mean)
    
    def get_stats(self):
        """
        Get statistics about the scheduler's performance.
        
        Returns:
            dict: Performance metrics
        """
        stats = self._stats.copy()
        if stats['placement_attempts'] > 0:
            stats['placement_success_rate'] = stats['placement_successes'] / stats['placement_attempts']
        else:
            stats['placement_success_rate'] = 0
        return stats


class BestFitScheduler(Scheduler):
    """
    A scheduler that places containers on the server with the least remaining resources
    that can still accommodate the request.
    
    This implementation aims to consolidate workloads and minimize resource fragmentation.
    """
    
    def __init__(self, env, cluster, idle_timeout_mean=2.0, idle_timeout_factor=1.0):
        """
        Initialize the scheduler.
        
        Args:
            env: SimPy environment
            cluster: Cluster instance containing servers
            idle_timeout_mean: Base mean value for exponentially distributed idle timeouts
            idle_timeout_factor: Factor to adjust timeouts based on system load
        """
        super().__init__(env, cluster)
        self.idle_timeout_mean = idle_timeout_mean
        self.idle_timeout_factor = idle_timeout_factor
        self._stats = {
            'placement_attempts': 0,
            'placement_successes': 0,
            'placement_failures': 0,
            'timeouts_set': 0
        }
    
    def find_server_for_spawn(self, request):
        """
        Find the server with the least available capacity that can still 
        accommodate the request (Best-Fit).
        
        Args:
            request: The request that needs a container
            
        Returns:
            Server instance or None if no suitable server is found
        """
        self._stats['placement_attempts'] += 1
        
        best_server = None
        min_remaining_capacity = float('inf')
        
        for server in self.cluster.servers:
            if server.has_capacity(request.cpu_demand, request.ram_demand, 
                                  request.cpu_warm, request.ram_warm):
                # Calculate remaining capacity as a percentage
                cpu_remaining_pct = server.cpu_reserve / server.cpu_capacity
                ram_remaining_pct = server.ram_reserve / server.ram_capacity
                
                # Take the average of CPU and RAM remaining
                avg_remaining = (cpu_remaining_pct + ram_remaining_pct) / 2
                
                if avg_remaining < min_remaining_capacity:
                    min_remaining_capacity = avg_remaining
                    best_server = server
        
        if best_server:
            self._stats['placement_successes'] += 1
        else:
            self._stats['placement_failures'] += 1
            
        return best_server
    
    def calculate_idle_timeout(self, container):
        """
        Calculate an exponentially distributed idle timeout with dynamic adjustment.
        
        The timeout is adjusted based on the current system load - shorter timeouts
        when the system is heavily loaded, longer timeouts when lightly loaded.
        
        Args:
            container: The container that has just become idle
            
        Returns:
            float: The timeout value in simulation time units
        """
        self._stats['timeouts_set'] += 1
        
        # Count total containers across all servers
        total_containers = sum(len(server.containers) for server in self.cluster.servers)
        
        # Calculate max possible containers based on CPU/RAM dimensions
        cpu_max = sum(server.cpu_capacity for server in self.cluster.servers) / container.cpu_alloc
        ram_max = sum(server.ram_capacity for server in self.cluster.servers) / container.ram_alloc
        max_possible = min(cpu_max, ram_max)
        
        # Adjust timeout based on system load
        if max_possible > 0:
            load_factor = total_containers / max_possible
            adjusted_mean = self.idle_timeout_mean * (1 - (load_factor * self.idle_timeout_factor))
            # Ensure the timeout doesn't go below a minimum threshold
            adjusted_mean = max(adjusted_mean, self.idle_timeout_mean * 0.1)
        else:
            adjusted_mean = self.idle_timeout_mean
            
        return random.expovariate(1.0/adjusted_mean)
    
    def get_stats(self):
        """
        Get statistics about the scheduler's performance.
        
        Returns:
            dict: Performance metrics
        """
        stats = self._stats.copy()
        if stats['placement_attempts'] > 0:
            stats['placement_success_rate'] = stats['placement_successes'] / stats['placement_attempts']
        else:
            stats['placement_success_rate'] = 0
        return stats