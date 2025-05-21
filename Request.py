class Request:
    """Represents a request arriving at the system."""
    def __init__(self, req_id, arrival_time, resource_demand, origin_node=None):
        self.cpu_warm = resource_demand['warm_cpu']
        self.ram_warm = resource_demand['warm_ram']
        self.cpu_demand = resource_demand['cpu_demand']
        self.ram_demand = resource_demand['ram_demand']
        self.cpu_warm_model = resource_demand['warm_cpu_model']
        self.ram_warm_model = resource_demand['warm_ram_model']
        self.resource_info = resource_demand
        self.id = req_id
        self.arrival_time = arrival_time
        self.start_service_time = -1 # Mark when service starts
        self.end_service_time = -1   # Mark when service ends
        self.origin_node = origin_node  # New: the topology node from which the request is sent
        # Initialize spawn time
        self.spawn_time = 0.0
        self.wait_time = 0.0   # Time waiting for container + assignment time
        self.state = "None"  # Initial state: "Pending", can change to "Running", "Rejected", "Finished"

    def __str__(self):
        return f"Req_{self.id}"
