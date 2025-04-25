class Request:
    """Represents a request arriving at the system."""
    def __init__(self, req_id, arrival_time, cpu_demand, ram_demand, warm_cpu, warm_ram, origin_node=None, app_id=None):
        self.cpu_warm = warm_cpu
        self.ram_warm = warm_ram
        self.id = req_id
        self.arrival_time = arrival_time
        self.cpu_demand = cpu_demand
        self.ram_demand = ram_demand
        self.start_service_time = -1 # Mark when service starts
        self.end_service_time = -1   # Mark when service ends
        self.origin_node = origin_node  # New: the topology node from which the request is sent
        # New: initialize spawn time and propagation delay
        self.spawn_time = 0.0
        self.prop_delay = 0.0
        self.app_id = app_id  # New: application ID this request belongs to

    def __str__(self):
        app_info = f" [App: {self.app_id}]" if self.app_id else ""
        return f"Req_{self.id} from {self.origin_node}{app_info}"
