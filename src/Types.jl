"""
    struct SimulationResult

Unified container for simulation output from PCE, Reachability, Scanning, or Monte Carlo.

# Fields:
- `sol`  : The raw solution object (e.g. ReachabilityAnalysis result, vector of ODESolutions, or PCE dict).
- `vars` : Tuple of variable symbols (used for plotting/indexing).
- `ts`   : Time vector (if applicable, otherwise `nothing`).
- `kind` : Dictionary containing metadata, must include `:type` key.
           Examples: `:pce`, `:reach`, `:bounded_reach`, `:scanning`, `:monte`
"""
struct SimulationResult
    sol
    vars
    ts
    kind::Dict{Symbol, Any}
end
