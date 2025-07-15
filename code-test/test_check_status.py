import simpy

# Define a simple Container class to simulate containers with a state
class Container:
    def __init__(self, state):
        self.state = state

# Define a class to manage the simpy.Store and count idle containers
class ContainerManager:
    def __init__(self, env):
        self.env = env
        self.idle_containers = simpy.Store(env)  # Create a simpy.Store

    def add_container(self, state):
        # Add a container with the given state to the store
        container = Container(state)
        self.idle_containers.put(container)

    def count_idle_containers(self):
        # Count containers with state "Idle"
        return sum(1 for container in self.idle_containers.items if container.state == "Idle")

# Test the functionality
def test_idle_count():
    # Initialize SimPy environment
    env = simpy.Environment()
    
    # Create a ContainerManager instance
    manager = ContainerManager(env)
    
    # Add containers with different states
    manager.add_container("Idle")
    manager.add_container("Busy")
    manager.add_container("Idle")
    manager.add_container("Idle")
    manager.add_container("Stopped")
    
    # Count idle containers
    idle_count = manager.count_idle_containers()
    
    # Print the result
    print(f"Number of Idle containers: {idle_count}")
    print(f"Expected number of Idle containers: 3")  # We added 3 Idle containers

# Run the test
if __name__ == "__main__":
    test_idle_count()