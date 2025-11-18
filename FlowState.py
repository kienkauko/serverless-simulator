class FlowState:
    """Tracks the state of an active transmission flow."""
    def __init__(self, request, path, packet_size, flow_type, process):
        self.request = request
        self.path = path
        self.packet_size = packet_size  # Total data to transmit (bits)
        self.flow_type = flow_type  # 'direct' or 'indirect'
        self.process = process  # The SimPy process handling this flow
        self.bytes_transmitted = 0  # Track progress
        self.start_time = None  # When transmission started
        self.last_update_time = None  # Last time we updated progress
        
    def get_remaining_data(self):
        """Calculate remaining data to transmit."""
        return max(0, self.packet_size - self.bytes_transmitted)
    
    def update_progress(self, current_time, bandwidth):
        """Update transmission progress based on elapsed time and bandwidth."""
        if self.last_update_time is not None:
            elapsed = current_time - self.last_update_time
            transmitted = bandwidth * elapsed
            self.bytes_transmitted += transmitted
        self.last_update_time = current_time