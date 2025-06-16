"""
    solve_parameter_scan(sys, tspan, grid_size; var_dict=Dict(), dt=0.01, interesting_vars=[])

Grid-scans parameter space and simulates all combinations.
Returns a `SimulationResult` with `:type => :scanning`.
"""
function solve_parameter_scan(sys, tspan, grid_size; var_dict=Dict(), dt=0.01, interesting_vars=[], solver = DE.Rodas5())
    param_ranges, not_intervals, intervals2 = getIntervals(sys, var_dict)
    ts = tspan[1]:dt:tspan[2]

    ranges_array = [range(IA.inf(v), IA.sup(v), length=grid_size) for v in values(param_ranges)]
    grid = collect(Base.Iterators.product(ranges_array...))
    all_states = [MTK.unknowns(sys)..., interesting_vars...]  # Changed to vector

    param_keys = collect(keys(param_ranges))
    prob2 = MTK.ODEProblem(sys, [], tspan, [])
    sols = Vector{MTK.ODESolution}(undef, length(grid))
    lc = ReentrantLock()
    n = length(grid)

    #println("Running scan over $n points using $(Threads.nthreads()) threads...")
    
    Threads.@threads for i in ProgressBars.ProgressBar(1:n)
        combo = grid[i]
        lock(lc)
        local_sys = deepcopy(sys)
        unlock(lc)
        p = Dict(param_keys[j] => combo[j] for j in 1:length(combo))
    
        prob = MTK.ODEProblem(local_sys, nothing, tspan, p, warn_initialize_determined = false)
        #prob = MTK.ODEProblem(sys, nothing, tspan, p, warn_initialize_determined = false)
        sol = DE.solve(prob, solver, saveat=ts)
        lock(lc)
        sols[i] = sol
        unlock(lc)
        #GC.gc()
    end

    return SimulationResult(
        sols,
        all_states,  # Corrected here
        ts,
        Dict(:type => :scanning, :grid_size => grid_size, :problem => sys)
    )
end



function solve_monte_carlo(sys, tspan, num_samples; var_dict=Dict(), dt=0.01, 
                            interesting_vars=[],
                            solver=DE.Rodas5(), 
                            verbose = true,
                            suppress_warn=true)
    interval_params, non_intervals, _ = getIntervals(sys, var_dict)
    ts = tspan[1]:dt:tspan[2]
    all_states = [MTK.unknowns(sys)..., interesting_vars...]

    param_keys = collect(keys(interval_params))
    dist_ranges = [Distributions.Uniform(IA.inf(v), IA.sup(v)) for v in values(interval_params)]
    kwargs = suppress_warn ? (; warn_initialize_determined=false) : NamedTuple()
    sols = Vector{MTK.ODESolution}(undef, num_samples)
    if verbose
        iter = ProgressBars.ProgressBar(1:num_samples)
    else
        iter = 1:num_samples
    end
    lc = ReentrantLock()
    samples = Vector{Any}(undef, num_samples)
    Threads.@threads for i in iter
        sampled_vals = ntuple(j -> rand(dist_ranges[j]), length(dist_ranges))
        param_sample = Dict(param_keys[j] => sampled_vals[j] for j in eachindex(param_keys))
        full_param = merge(param_sample, non_intervals)
        lock(lc)
        syss = deepcopy(sys)
        unlock(lc)
        prob = MTK.ODEProblem(syss, nothing, tspan, full_param; kwargs...)
        sol = DE.solve(prob, solver,saveat=ts)
        
        #lock(lc)
        sols[i] = sol
        samples[i] = param_sample
        #unlock(lc)
    end

        return SimulationResult(
        sols,
        all_states,  # Corrected here
        ts,
        Dict(:type => :monte, :num_samples => num_samples, :problem => sys, :samples => samples)
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
function compute_bounds(result::SimulationResult; idxs=1, verbose = true)
    sols = result.sol
    all_vars = result.vars
    times = result.ts
    kind = result.kind[:type]
    n = length(times)
    num_samples = length(sols)

    # Normalize inputs
    idxlist =
        isa(idxs, Tuple) && length(idxs) == 2 ? collect(idxs) :
        isa(idxs, AbstractVector) ? idxs :
        [idxs]

    selected_vars = Vector{Any}(undef, length(idxlist))
    bounds = Dict{Symbol, Vector{Float64}}()
    if verbose
        pbar = ProgressBars.ProgressBar(total=length(idxlist)*num_samples)
    end
    lc = ReentrantLock()
    for (i, idx) in enumerate(idxlist)
        var = idx isa Int ? all_vars[idx] : idx
        selected_vars[i] = var

        minv = fill(Inf, n)
        maxv = fill(-Inf, n)

        
        locsol = Vector{Vector{Float64}}(undef, num_samples)
        Threads.@threads for k in 1:num_samples
            #lock(lc)
            #sol1=deepcopy(sols[k])
            #unlock(lc)
            #locsol[k] = sol1[var]
            locsol[k] = sols[k][var]
            if verbose 
                lock(lc)
                ProgressBars.update(pbar)
                unlock(lc)
            end
            #k%100==0 && GC.gc()
        end
        #println(typeof(sols))
        for t in 1:n
            #println("here")
            vals = [sol[t] for sol in locsol]
            minv[t] = minimum(vals)
            maxv[t] = maximum(vals)
        end

        bounds[Symbol("min$i")] = minv
        bounds[Symbol("max$i")] = maxv
    end
    GC.gc()
    return SimulationResult(
        bounds,
        selected_vars,  # Corrected here
        times,
        merge(result.kind, Dict(:type => Symbol("$(kind)_bounded"), :idxs => idxlist))
    )
end
