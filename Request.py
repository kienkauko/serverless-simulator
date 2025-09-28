class Request:
    """Represents a request arriving at the system."""
    def __init__(self, req_id, arrival_time, resource_demand, origin_node=None, data_node=None, app_id=None):
        self.id = req_id
        self.arrival_time = arrival_time
        self.cpu_warm = resource_demand["cpu_warm"]
        self.ram_warm = resource_demand["ram_warm"]
        self.cpu_demand = resource_demand["cpu_demand"]
        self.ram_demand = resource_demand["ram_demand"]
        self.bandwidth_direct = resource_demand["bandwidth_direct"] # bw from user
        self.bandwidth_indirect = resource_demand["bandwidth_indirect"] # bw from data 
        self.packet_size_direct_upload = resource_demand["packet_size_direct_upload"] # in bits, default to 1500 bytes (MTU)
        self.packet_size_direct_download = resource_demand["packet_size_direct_download"] # in bits, default to 1500 bytes (MTU)
        self.packet_size_indirect_upload = resource_demand["packet_size_indirect_upload"] # in bits, default to 1500 bytes (MTU)
        self.packet_size_indirect_download = resource_demand["packet_size_indirect_download"] # in bits, default to 1500 bytes (MTU)
        self.data_path_required = resource_demand["data_path_required"]  # Whether data path is required
        self.data_node = data_node
        self.start_service_time = -1 # Mark when service starts
        self.end_service_time = -1   # Mark when service ends
        self.origin_node = origin_node  # New: the topology node from which the request is sent
        self.local_edge_cluster = None  # New: the local edge cluster if applicable
        # New: initialize spawn time and propagation delay
        self.spawn_time = 0.0
        self.prop_delay = 0.0
        self.network_delay = 0.0
        self.processing_time = 0.0  # Track processing time for this request
        # Store potential propagation delays for each cluster
        self.potential_prop_delays = {}
        self.app_id = app_id  # New: application ID this request belongs to
        self.assigned_cluster = None  # Track which cluster the request is assigned to
        self.waiting_time = 0.0  # Track the waiting time for this request
        self.waiting_start_time = -1  # Mark when waiting starts
        self.state = "Pending"  # Track request state: "Pending", "Running", or "Finished"
       
        self.max_trans_delay = -1
        self.bottleneck = None  # Track bottleneck link for direct path

        # self.delay_by_level_direct = None  # Track delay by hierarchy level for direct path
        # self.delay_by_level_indirect = None  # Track delay by hierarchy level for indirect path

    def __str__(self):
        app_info = f" [App: {self.app_id}]" if self.app_id else ""
        cluster_info = f" [Cluster: {self.assigned_cluster}]" if self.assigned_cluster else ""
        # state_info = f" [State: {self.state}]"
        return f"Req_{self.id} from {self.origin_node}{app_info}{cluster_info}"
    