class FlowState:
    """Tracks the state of an active transmission flow."""
    def __init__(self, env, request, path, data_size, flow_type):
        self.request = request
        self.path = path
        self.path_links = frozenset((path[i], path[i+1]) for i in range(len(path) - 1))
        self.data_size = data_size  # Total data to transmit (bits)
        self.flow_type = flow_type  # 'direct' or 'indirect'
        self.data_remaining = data_size  # Remaining data to transmit
        self.data_transmitted = 0  # Track progress
        self.start_time = None  # When transmission started
        self.last_update_time = None  # Last time we updated progress
        self.allocated_bandwidth = None  # Minimum bandwidth across path (bottleneck)
        self.bottleneck_level = None  # Hierarchy level of bottleneck link
        self.bottleneck_exact_link = None  # Exact link that is the bottleneck
        self.completion_event = env.event()  # Triggered when flow completes

    def update_progress(self, current_time):
        """Update transmission progress based on elapsed time and bandwidth."""
        if self.allocated_bandwidth and self.last_update_time is not None:
            elapsed = current_time - self.last_update_time
            if elapsed > 0:
                transmitted = self.allocated_bandwidth * elapsed
                self.data_transmitted += transmitted
                self.data_remaining = max(0, self.data_size - self.data_transmitted)
        self.last_update_time = current_time
