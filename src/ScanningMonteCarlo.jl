"""
    solve_parameter_scan(sys, tspan, grid_size; var_dict=Dict(), dt=0.01, interesting_vars=[])

Grid-scans parameter space and simulates all combinations.
Returns a `SimulationResult` with `:type => :scanning`.
"""
function solve_parameter_scan(sys, tspan, grid_size; var_dict=Dict(), dt=0.01, interesting_vars=[])
    param_ranges = getIntervals(sys, var_dict)
    ts = tspan[1]:dt:tspan[2]

    ranges_array = [range(IA.inf(v), IA.sup(v), length=grid_size) for v in values(param_ranges)]
    grid = collect(Base.Iterators.product(ranges_array...))
    all_states = [MTK.unknowns(sys)..., interesting_vars...]  # Changed to vector

    param_keys = collect(keys(param_ranges))
    prob = MTK.ODEProblem(sys, [], tspan, [])
    sols = Vector{MTK.ODESolution}(undef, length(grid))
    for (i, combo) in ProgressBars.ProgressBar(enumerate(grid))
        p = Dict(param_keys[j] => combo[j] for j in 1:length(combo))
        prob = MTK.ODEProblem(sys, nothing, tspan, p, warn_initialize_determined = false)
        sols[i] = DifferentialEquations.solve(prob, saveat=ts)
    end

    return SimulationResult(
        sols,
        all_states,  # Corrected here
        ts,
        Dict(:type => :scanning, :grid_size => grid_size, :problem => sys)
    )
end

"""
    solve_monte_carlo(sys, tspan, num_samples; var_dict=Dict(), dt=0.01, interesting_vars=[])

Randomly samples parameters from intervals. Returns a `SimulationResult` with `:type => :monte`.
"""
function solve_monte_carlo(sys, tspan, num_samples; var_dict=Dict(), dt=0.01, interesting_vars=[])
    param_ranges = getIntervals(sys, var_dict)
    ts = tspan[1]:dt:tspan[2]
    all_states = [MTK.unknowns(sys)..., interesting_vars...]  # Changed to vector

    sols = Vector{MTK.ODESolution}(undef, num_samples)
    keys_ = collect(keys(param_ranges))
    prob = MTK.ODEProblem(sys, [], tspan, [])
    #Should be possible to multithread but cant have shared sys I think
    for i in ProgressBars.ProgressBar(1:num_samples)
        sampled = Dict(k => rand(Distributions.Uniform(IA.inf(v),IA.sup(v))) for (k, v) in param_ranges)
        prob = MTK.ODEProblem(sys, nothing, tspan, sampled, warn_initialize_determined = false)
        sols[i] = DifferentialEquations.solve(prob, saveat=ts)
    end

    return SimulationResult(
        sols,
        all_states,  # Corrected here
        ts,
        Dict(:type => :monte, :num_samples => num_samples, :problem => sys)
    )
end

"""
    compute_bounds(result::SimulationResult; idxs)

Computes envelope bounds over time or 2D slices for scanning or Monte Carlo solutions.

Accepts:
- A single index or symbolic variable
- A tuple of two variables (for phase plots)
- A list of variables or indices

Returns:
- A single `SimulationResult` with `.sol = Dict(:min1, :max1, :min2, :max2, ...)`
- `.vars` lists the included variables
- `.kind[:idxs]` remembers which were requested
"""
function compute_bounds(result::SimulationResult; idxs=1)
    sols = result.sol
    all_vars = result.vars
    times = result.ts
    kind = result.kind[:type]
    n = length(times)

    # Normalize inputs
    idxlist =
        isa(idxs, Tuple) && length(idxs) == 2 ? collect(idxs) :
        isa(idxs, AbstractVector) ? idxs :
        [idxs]

    selected_vars = Vector{Any}(undef, length(idxlist))
    bounds = Dict{Symbol, Vector{Float64}}()

    for (i, idx) in enumerate(idxlist)
        var = idx isa Int ? all_vars[idx] : idx
        selected_vars[i] = var

        minv = fill(Inf, n)
        maxv = fill(-Inf, n)

        for t in 1:n
            vals = [sol[var][t] for sol in sols]
            minv[t] = minimum(vals)
            maxv[t] = maximum(vals)
        end

        bounds[Symbol("min$i")] = minv
        bounds[Symbol("max$i")] = maxv
    end

    return SimulationResult(
        bounds,
        selected_vars,  # Corrected here
        times,
        merge(result.kind, Dict(:type => Symbol("$(kind)_bounded"), :idxs => idxlist))
    )
end
