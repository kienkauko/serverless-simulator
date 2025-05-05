import simpy

class Container:
    def __init__(self, state):
        self.state = state
    
    def __repr__(self):
        return f"Container(state={self.state})"

class System:
    def __init__(self, env):
        self.env = env
        self.idle_containers = simpy.Store(env)  # FIFO queue for containers
    
    def get_idle(self):
        count = 0
        while True:
            get_op = self.idle_containers.get()
            container = yield get_op
            print(f"count: {count}")
            count += 1
            if container.state != "Idle":
                print(f"WARNING: {self.env.now:.2f} - Retrieved container {container} is not idle, but {container.state}. Discarding it.")
                continue
            print(f"SUCCESS: {self.env.now:.2f} - Retrieved idle container {container} by process.")
            return container  # Exit generator with Idle container
    
    def main(self):
        # Start three processes with delays
        process_1 = self.env.process(self.get_idle())
        yield self.env.timeout(1)
        process_2 = self.env.process(self.get_idle())
        yield self.env.timeout(1)
        process_3 = self.env.process(self.get_idle())
        yield self.env.timeout(1)
        
        # Add containers to idle_containers
        container = Container("Dead")
        self.idle_containers.put(container)
        print(f"{self.env.now:.2f} - Added {container}")
        container = Container("Dead")
        self.idle_containers.put(container)
        print(f"{self.env.now:.2f} - Added {container}")
        container = Container("Idle")
        self.idle_containers.put(container)
        print(f"{self.env.now:.2f} - Added {container}")

# Create simulation environment
env = simpy.Environment()
system = System(env)

# Run the main process
env.process(system.main())

# Run the simulation
env.run()