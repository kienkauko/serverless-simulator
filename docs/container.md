# Container (`Container.py`)

A `Container` object represents a runtime environment for executing requests of a specific application (`app_id`). It acts as a stateful entity that is spawned on a `Server`, consumes resources, serves requests, and can be kept "warm" in an idle state for a period before being terminated.

## 1. Container States and Resources

A container can be in one of several states: `Assigned`, `Active`, `Idle`, or `Dead`. Its resource consumption changes based on its state.

-   **Warm/Idle State:** When a container is created or has finished a request, it is `Idle`. In this state, it consumes a baseline amount of resources to stay ready for the next request.
-   **Active State:** When a request is assigned and being processed, the container becomes `Active`. It scales up its resource usage to meet the demands of the request.

These two levels of resource consumption are defined during the container's initialization:

```python
# In Container.__init__
# Warm (Idle) resource allocation
self.cpu_alloc = request.cpu_warm
self.ram_alloc = request.ram_warm

# Active (Serving) resource reservation/limit
self.cpu_reserve = request.cpu_demand
self.ram_reserve = request.ram_demand
```

-   `cpu_alloc` / `ram_alloc`: The **current** amount of resources the container is using from the server.
-   `cpu_reserve` / `ram_reserve`: The amount of resources the container will scale up to when it becomes `Active`. This is analogous to the "limit" concept in Kubernetes.

When a container is idle, it is added to the `system.app_idle_containers` dictionary, making it available for the `LoadBalancer` to assign new requests to it.

## 2. Container Lifecycle and Resource Management

The container's lifecycle is managed through a series of `simpy` processes that define its behavior from serving a request to being terminated. Resource allocation and deallocation are tied directly to these lifecycle events.

1.  **`service_lifecycle()`**: This is the main process for handling an active request.
    -   It first calls **`scale_for_request()`** to acquire the additional resources needed to go from warm to active. This function calculates the difference (`delta`) between the active demand and the current warm allocation and requests it from the `Server`.
        ```python
        # In scale_for_request()
        delta_cpu = request.cpu_demand - self.cpu_alloc
        delta_ram = request.ram_demand - self.ram_alloc
        # ...
        self.server.allocate_resources(delta_cpu, delta_ram)
        ```
    -   It then simulates the request's processing time (`yield self.env.timeout(service_time)`).
    -   After this process completes, the `System` calls `release_request()`.

2.  **`release_request()`**: This method transitions the container from `Active` back to `Idle`.
    -   It scales resources **down** from active to warm. It calculates the `delta` by subtracting the warm resources from the current allocation and returns those resources to the `Server`.
        ```python
        # In release_request()
        delta_cpu = self.cpu_alloc - self.current_request.cpu_warm
        delta_ram = self.ram_alloc - self.current_request.ram_warm
        # ...
        self.server.cpu_real += delta_cpu
        self.server.ram_real += delta_ram
        ```
    -   It then starts the `idle_lifecycle()` process to manage the container's idle period.

3.  **`idle_lifecycle()`**: This process determines how long the container stays warm.
    -   It waits for a specific `self.time_out` duration.
    -   If the timeout completes without being interrupted, the container is terminated by calling **`release_resources()`**. This function releases **all** remaining (warm) resources back to the server.
        ```python
        # In release_resources()
        self.server.cpu_real += self.cpu_alloc
        self.server.ram_real += self.ram_alloc
        self.server.cpu_reserve += self.cpu_reserve
        self.server.ram_reserve += self.ram_reserve
        ```
    -   If a new request is assigned to the container before the timeout, this process is **interrupted**, and the container goes back to the `service_lifecycle`.
