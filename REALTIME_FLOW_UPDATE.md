# Real-Time Flow Update Implementation

## Overview

This implementation makes network delay calculation more realistic by dynamically adjusting bandwidth allocation when flows are added or removed from shared links. Previously, the simulation calculated delay once at the beginning and assumed it remained constant. Now, the system continuously monitors network state and recalculates delays when conditions change.

## Key Changes

### 1. FlowState Class (New)
A new class that tracks the state of each active transmission flow:

```python
class FlowState:
    - request: The associated request
    - path: The network path being used
    - packet_size: Total data to transmit (bits)
    - flow_type: 'direct' or 'indirect'
    - process: The SimPy process handling this flow
    - bytes_transmitted: Tracks transmission progress
    - start_time: When transmission started
    - last_update_time: Last time progress was updated
```

**Methods:**
- `get_remaining_data()`: Returns remaining data to transmit
- `update_progress(current_time, bandwidth)`: Updates transmission progress based on elapsed time and allocated bandwidth

### 2. Enhanced Topology Class

#### New Attributes
- `active_flows`: Dictionary tracking all active transmission processes per link
  - Format: `{(node1, node2): [list of FlowState objects]}`

#### Refactored: `update_request_delay()`
Now an **interruptible SimPy process** that:

1. **Initial Setup** (one-time):
   - Calculates static propagation delay
   - Calculates TCP overhead (connection setup + slow start)
   - Creates FlowState objects for direct and indirect paths
   - Registers flows with the network

2. **Transmission Loop** (interruptible):
   - Calculates expected delay based on current bandwidth allocation
   - Identifies bottleneck links
   - Yields timeout for expected transmission time
   - **On Interruption**: Updates transmission progress and recalculates delay with new bandwidth allocation
   - Continues until all data is transmitted

3. **Cleanup**:
   - Unregisters flows from the network
   - Interrupts other flows sharing the same links (their bandwidth just increased)
   - Updates request statistics

#### New Helper Methods

**`_register_flow(flow_state)`**
- Registers a flow with all links in its path
- Adds flow to `active_flows` registry
- Increments `num_active_flows` counter on each link

**`_unregister_flow(flow_state)`**
- Removes a flow from all links in its path
- Removes flow from `active_flows` registry
- Decrements `num_active_flows` counter on each link

**`_interrupt_affected_flows(path)`**
- Finds all flows sharing links with the given path
- Interrupts those flows so they can recalculate their bandwidth allocation

**`_get_allocated_bandwidth(flow_state)`**
- Calculates fair-share bandwidth for a flow
- Returns the minimum per-flow bandwidth across all links in the path (bottleneck)

**`_calculate_flow_delay(flow_state)`**
- Calculates expected transmission delay based on:
  - Remaining data to transmit
  - Current bandwidth allocation (fair sharing)
  - Bottleneck link identification
- Returns tuple: (expected_delay, bottleneck_level)

#### Updated: `make_paths()` and `remove_paths()`
These methods are now **no-ops** (kept for backward compatibility):
- Flow registration/unregistration is handled automatically in `update_request_delay()`
- Flow counting is done during `_register_flow()` and `_unregister_flow()`

### 3. Updated System Class

Removed redundant `make_paths()` and `remove_paths()` calls in `handle_request()`:
- Before upload: ~~`make_paths()` → `update_request_delay()` → `remove_paths()`~~
- Now: `update_request_delay()` (handles everything)

## How It Works

### Scenario Example

1. **Flow A starts** transmitting on path [N1 → N2 → N3]
   - Registers with links (N1,N2) and (N2,N3)
   - Gets full bandwidth: 100 Mbps
   - Expects to finish in 10 seconds

2. **Flow B starts** (5 seconds later) on path [N1 → N2 → N4]
   - Registers with links (N1,N2) and (N2,N4)
   - **Interrupts Flow A** (they share link N1→N2)
   
3. **Flow A recalculates**:
   - Transmitted 50% of data (5 seconds × 100 Mbps)
   - Remaining: 50% of data
   - New bandwidth: 50 Mbps (fair share with Flow B)
   - New expected time: 10 seconds (50% data ÷ 50 Mbps)

4. **Flow B calculates**:
   - Link (N1,N2): 50 Mbps (shared)
   - Link (N2,N4): 100 Mbps (exclusive)
   - **Bottleneck**: 50 Mbps on (N1,N2)
   - Expected time: based on 50 Mbps

5. **Flow A completes** (15 seconds total)
   - Unregisters from links
   - **Interrupts Flow B** (bandwidth on N1→N2 just increased)

6. **Flow B recalculates**:
   - New bandwidth on (N1,N2): 100 Mbps (no longer shared)
   - Continues with improved bandwidth

## Benefits

1. **Realistic Network Modeling**: Bandwidth is dynamically shared based on actual network load
2. **Fair Sharing**: All flows get equal bandwidth on congested links
3. **Accurate Delay Calculation**: Accounts for changing network conditions
4. **Bottleneck Detection**: Correctly identifies which link limits performance
5. **Event-Driven**: Only recalculates when network state changes (efficient)

## Technical Details

### Interruption Handling
- Uses SimPy's `Interrupt` mechanism
- Each flow tracks its progress (bytes transmitted)
- On interruption:
  - Updates progress based on elapsed time and old bandwidth
  - Recalculates remaining work
  - Recalculates delay with new bandwidth allocation

### Fair Sharing Model
- Bandwidth is divided equally among all active flows on a link
- Each flow gets: `link_capacity / num_active_flows`
- Flow's actual bandwidth = minimum across all links (bottleneck)

### Performance Considerations
- Flows only interrupt each other when sharing links
- Progress updates are O(path_length)
- Interruption checking is O(flows_per_link × links_per_path)
- Efficient for typical network topologies

## Testing Recommendations

1. **Single Flow**: Verify no interruptions, completes in expected time
2. **Two Overlapping Flows**: Verify fair sharing and recalculation
3. **Sequential Flows**: Verify no interference when not overlapping
4. **Multiple Paths**: Verify independent paths don't affect each other
5. **Cascading Effects**: Start many flows, verify correct bandwidth division

## Future Enhancements

Possible improvements:
1. **Weighted Fair Queuing**: Give different priorities to flows
2. **TCP Congestion Control**: Model slow start, congestion avoidance
3. **Packet Loss**: Add loss probability based on congestion
4. **Queue Delays**: Model queuing delay at routers
5. **Burst Traffic**: Handle non-uniform traffic patterns

## Migration Notes

- **Backward Compatible**: Old code continues to work
- `make_paths()` and `remove_paths()` can be safely removed from calling code
- Statistics collection remains unchanged
- Request objects have same attributes
