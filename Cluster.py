from Server import Server

class Cluster:
    def __init__(self, env, node, num_servers, server_cpu, server_ram):
        self.env = env
        self.node = node  # Topology node where this cluster is located
        self.servers = [Server(env, i, server_cpu, server_ram) for i in range(num_servers)]
