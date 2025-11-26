class FlowState:
    """Tracks the state of an active transmission flow."""
    def __init__(self, request, path, data_size, flow_type, process):
        self.request = request
        self.path = path
        # self.changed_links = []  # Links that caused interruptions
        self.data_size = data_size  # Total data to transmit (bits)
        self.flow_type = flow_type  # 'direct' or 'indirect'
        self.process = process  # The SimPy process handling this flow
        self.data_remaining = data_size  # Remaining data to transmit
        self.data_transmitted = 0  # Track progress
        self.start_time = None  # When transmission started
        self.last_update_time = None  # Last time we updated progress
        self.min_bandwidth = None  # Cached minimum bandwidth
        self.bottleneck_level = None  # Cached bottleneck level
        self.bottleneck_exact_link = None  # Cached bottleneck link
    # def get_remaining_data(self):
    #     """Calculate remaining data to transmit."""
    #     self.data_remaining = max(0, self.data_size - self.data_transmitted)
    #     return self.data_remaining
    
    def update_progress(self, current_time):
        """Update transmission progress based on elapsed time and bandwidth."""
        elapsed = current_time - self.last_update_time
        transmitted = self.min_bandwidth * elapsed
        self.data_transmitted += transmitted
        self.data_remaining = max(0, self.data_size - self.data_transmitted)
        self.last_update_time = current_time