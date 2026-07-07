# --- Simulation Configuration ---

config = {
    "system": {
        "num_servers": 15,
        "sim_time": 10000,
        "warmup_time": 1000,       # Transient warm-up period; metrics collected only after this
        "verbose": False,
        "warm_percent": 0,       # Fraction of total container slots pre-warmed
    },
    "distribution": {
        "spawn-distribution": "exponential",   # Options: "deterministic", "lognormal", "exponential"
        "arrival-distribution": "exponential", # Options: "deterministic", "weibull", "exponential"
        "service-distribution": "exponential",      # Options: "exponential", "deterministic", "traces"
        "idle-distribution": "deterministic",  # Options: "deterministic", "markov" (exponential idle timer)
    },
    "server": {
        "cpu_capacity": 100.0,     # % of total CPU
        "ram_capacity": 100.0,     # % of total RAM
        "peak_power": 150.0,       # Peak power consumption in Watts
        "power_scale": 0.4,        # Idle power fraction (0–1)
    },
    "request": {
        "arrival_rate_mean": 2.35,   # Average requests per time unit (λ)
        "arrival_rate_std": 0,     # Std-dev of arrival rate (used by Weibull)
        "service_rate": 1/2.12,        # Average service completions per time unit (μ)
        # Resource demands per request
        "warm_cpu": 0.05,
        "warm_ram": 2.60,
        "cold_start_cpu": 5.35,
        "cold_start_ram": 1.51,
        "cpu_demand": 7.03,
        "ram_demand": 2.62,
    },
    "container": {
        "spawn_time_mean": 10,   # Mean cold-start spawn duration (time units)
        "spawn_time_std": 0.46,    # Std-dev of spawn duration
        "idle_cpu_timeout": 150,     # Idle eviction timeout (0 = disabled for warm pool)
    },
}
