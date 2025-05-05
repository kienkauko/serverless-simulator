from Server import Server

class Cluster:
    def __init__(self, env, num_servers, server_cpu, server_ram):
        self.env = env
        self.servers = [Server(env, i, server_cpu, server_ram) for i in range(num_servers)]
