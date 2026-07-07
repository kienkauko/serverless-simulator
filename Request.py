class Request:
    """Represents a request arriving at the system."""

    def __init__(self, req_id, arrival_time, resource_demand):
        self.id = req_id
        self.arrival_time = arrival_time
        self.cpu_warm = resource_demand['warm_cpu']
        self.ram_warm = resource_demand['warm_ram']
        self.cpu_cold_start = resource_demand['cold_start_cpu']
        self.ram_cold_start = resource_demand['cold_start_ram']
        self.cpu_demand = resource_demand['cpu_demand']
        self.ram_demand = resource_demand['ram_demand']
        self.resource_info = resource_demand
        self.start_service_time = -1
        self.end_service_time = -1
        self.spawn_time = 0.0
        self.wait_time = 0.0
        self.state = "Pending"

    def __str__(self):
        return f"Req_{self.id}"
