# Load Balancer (LB) and Scheduler — Concise Guide

**Note:** The current `LoadBalancer` and `Scheduler` code is messy and contains several unused functions and placeholders for future strategies. This guide focuses only on the actual control flow used by the `System` to handle requests.

## 1. Entry Point: `System.handle_request`

When a request is generated, the `System` orchestrates its lifecycle.

1.  **Find Viable Clusters:** The `System` first calls `topology.find_cluster(request)` to get a list of `target_clusters` that can potentially handle the request. This list is ordered based on network topology (e.g., by propagation delay).
2.  **Delegate to Load Balancer:** The `System` then delegates the core assignment logic to the `LoadBalancer`:
    ```python
    # In System.py
    handle_request_process = self.load_balancer.handle_request(request, target_clusters)
    assignment_result, container, cluster = yield self.env.process(handle_request_process)
    ```

## 2. Load Balancer: `LoadBalancer.handle_request`

The `LoadBalancer` uses a **"greedy"** strategy (`_handle_request_greedy`) to find a container for the request.

1.  **Iterate Through Clusters:** It iterates through the `target_clusters` list in the given order.
2.  **Check for Idle Container:** For each cluster, it checks if an idle container for the specific application (`request.app_id`) exists in the `system.app_idle_containers` pool.
3.  **Use Idle Container:** If an idle container is found, it is immediately assigned to the request via `assign_request_to_container`. This function also cancels the container's pending removal timeout. The process succeeds and stops here.
4.  **Spawn New Container:** If no idle containers are available in the current cluster, the `LoadBalancer` asks that cluster's dedicated `Scheduler` to create a new one:
    ```python
    # In LoadBalancer.py
    scheduler = self.schedulers[cluster_name]
    spawn_result, container = yield self.env.process(scheduler.spawn_container_for_request(request, self.system))
    ```
5.  **Fallback and Block:**
    *   If the scheduler successfully spawns a container (`spawn_result` is `True`), the request is assigned, and the process succeeds.
    *   If the scheduler fails (e.g., no server capacity), the `LoadBalancer` moves to the **next cluster** in the list and repeats the process.
    *   If all clusters in the list fail to provide a container, the request is **blocked**.

## 3. Scheduler: `Scheduler.spawn_container_for_request`

Each cluster has its own `Scheduler` instance. Its job is to find a server within its cluster to host a new container. The current strategy is **`FirstFitScheduler`**.

1.  **Find a Server:** The scheduler calls `find_server_for_spawn(request)`. The `FirstFitScheduler` implementation iterates through its list of servers and returns the **first one** that has enough CPU and RAM.
2.  **Initiate Spawning:** If a server is found, the scheduler tells that server to start the container spawning process, which is a `simpy` process that simulates the time delay:
    ```python
    # In Scheduler.py
    container = yield self.env.process(server.spawn_container_process(system, request))
    ```
3.  **Return Result:** The scheduler returns `(True, container)` on success or `(False, None)` on failure, which the `LoadBalancer` uses to continue its logic.



