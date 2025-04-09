"""
    solve_reachability(data::Tuple, tspan; kwargs...)

Overload that accepts the full output of `createIVP(...)` directly.
Extracts the IVP and vars automatically.
"""
function solve_reachability(data::Tuple, tspan; dt=0.01, solver=RA.TMJets21b())
    ivp, vars, _ = data
    return solve_reachability(ivp, vars, tspan; dt=dt, solver=solver)
end

"""
    solve_reachability(ivp, vars, tspan, dt; solver=RA.TMJets21b())

Solves a reachability problem using TaylorModels. Returns a `SimulationResult`
with `:type => :reach`.
"""
function solve_reachability(ivp, vars, tspan; dt=0.01, solver=RA.TMJets21b())
    println(solver)
    sol = RA.solve(ivp, tspan=tspan, alg=solver)
    ts = tspan[1]:dt:tspan[2]
    
    return SimulationResult(
        sol,
        vars,
        ts,
        Dict(
            :type => :reach,
            :solver => solver,
            :problem => ivp,
            :dt => dt
        )
    )
end
"""
    compute_bounds(result::SimulationResult; idxs)

Computes envelope bounds over time or 2D slices for scanning or Monte Carlo solutions.

Accepts:
- A single index or symbolic variable
- A tuple of two variables (for phase plots)
- A list of variables or indices

Returns a single `SimulationResult` with:
- `sol`: a dictionary with keys like `:min1`, `:max1`, `:min2`, `:max2`, etc.
- `vars`: the selected variables (as they were passed or extracted)
- `ts`: the time vector
- `kind`: updated metadata including `:type => :*_bounded` and `:idxs` holding the selected indices.
"""
function get_bounds(result::SimulationResult; idxs=1:length(result.vars))
    sol = result.sol
    vars = result.vars

    # Normalize indices
    actual_indices = [i isa Int ? i : findfirst(x -> isequal(x, i), vars) for i in idxs]

    # Get the flowpipe, either via `.F.Xk` (MixedFlowpipe) or `.Xk` (Flowpipe)
    Xk = hasproperty(sol, :F) ? sol.F.Xk : sol.Xk

    # Compute interval bounds over time
    bounded_sol = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()

    for j in actual_indices
        lb_vec = Float64[]
        ub_vec = Float64[]
        for x in 1:length(Xk)
            intval = TaylorModels.quadratic_fast_bounder(
                RA.evaluate(Xk[x], Xk[x].Δt)[j]
            )
            push!(lb_vec, RA.inf(intval))
            push!(ub_vec, RA.sup(intval))
        end
        bounded_sol[j] = (lb_vec, ub_vec)
    end

    # Compute time vector
    save_times = [
        0.5 * (RA.sup(Xk[x].Δt[1]) + RA.inf(Xk[x].Δt[1]))
        for x in 1:length(Xk)
    ]

    new_kind = merge(result.kind, Dict(:type => :bounded_reach, :indices => actual_indices))
    return SimulationResult(bounded_sol, actual_indices, save_times, new_kind)
end



function split_intervals_reach(ivp::RA.InitialValueProblem, var_splits::Dict{Int, Int})
    base_u = ivp.x0  # This is an IntervalBox
    configs = [base_u]

    for (idx, n) in var_splits
        new_configs = []
        iv = base_u[idx]
        lb = RA.inf(iv)
        ub = RA.sup(iv)
        Δ = (ub - lb) / n

        for cfg in configs
            for i in 0:(n-1)
                intervals = Tuple(cfg)
                intervals_new = Base.setindex(intervals, IA.Interval(lb + i*Δ, lb + (i+1)*Δ), idx)
                cfg_new = IA.IntervalBox(intervals_new...)
                push!(new_configs, cfg_new)
            end
        end

        configs = new_configs
    end

    return configs
end

function solve_reachability_split(ivp::RA.InitialValueProblem, vars, tspan;
    dt=0.01,
    solver=RA.TMJets21b(adaptive=false),
    split_counts=Dict(),
    threading = true
)
    if isempty(split_counts)
        return solve_reachability(ivp, vars, tspan; dt=dt, solver=solver)
    end

    # Generate split initial sets
    u0_configs = split_intervals_reach(ivp, split_counts)

    # Construct a new IVP with multiple initial boxes
    X0 = convert(Vector{IA.IntervalBox{length(u0_configs[1]), Float64}}, u0_configs)

    order = length(X0[1])  # assuming all have same dim
    f = ivp.s.f

    # Build a new IVP directly with multiple initial conditions
    prob = @ivp(x' = f(x), dim = order, x(0) ∈ X0)

    # Use ReachabilityAnalysis' built-in multi-IVP solver
    sol = RA.solve(prob, tspan=tspan, alg=solver, threading=threading)

    # Wrap each solution into a SimulationResult to match your structure
    ts = tspan[1]:dt:tspan[2]
    sub_results = [
        SimulationResult(s, vars, ts, Dict(:type => :reach, :solver => solver, :dt => dt))
        for s in sol
    ]

    return SimulationResult(sub_results, vars, ts, Dict(:type => :split_reach, :split_counts => split_counts))
end


function calculate_bounds_split_reach(split_res::SimulationResult; idxs=1:length(split_res.vars))
    if split_res.kind[:type] != :split_reach
        error("Expected a split_reach SimulationResult")
    end

    results = split_res.sol   # array of SimulationResult for each sub-run
    ts = split_res.ts         # unified time vector
    bounded = [get_bounds_interpolated(r; idxs=idxs, new_ts=ts) for r in results]

    final_bounds = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()

    for i in idxs
        lowers = [b.sol[i][1] for b in bounded]
        uppers = [b.sol[i][2] for b in bounded]
        num_t = length(ts)
        min_bound = [minimum(getindex.(lowers, t)) for t in 1:num_t]
        max_bound = [maximum(getindex.(uppers, t)) for t in 1:num_t]
        final_bounds[i] = (min_bound, max_bound)
    end

    new_kind = merge(split_res.kind, Dict(:type => :bounded_reach_split, :origin => :split, :indices => idxs))
    return SimulationResult(
        final_bounds,
        idxs,
        ts,
        new_kind
    )
end



function linear_interp(tq, T, Y)
    # Basic 1D linear interpolation.
    # T and Y must be the same length, T is sorted.
    # If tq is out of [T[1], T[end]], we clamp to endpoints.
    if tq <= T[1]
        return Y[1]
    elseif tq >= T[end]
        return Y[end]
    else
        idx = searchsortedlast(T, tq)
        # T[idx], T[idx+1]
        t1, t2 = T[idx], T[idx+1]
        y1, y2 = Y[idx], Y[idx+1]
        return y1 + (tq - t1)*(y2 - y1)/(t2 - t1)
    end
end

"""
    get_bounds_interpolated(subres::SimulationResult; idxs, new_ts)

Calls `get_bounds(subres; idxs=idxs)`, then linearly interpolates
the bounding arrays onto the specified `new_ts`.

Returns a SimulationResult in the same format, but the bounding
arrays all match `length(new_ts)`.
"""
function get_bounds_interpolated(subres::SimulationResult; idxs, new_ts)
    raw = get_bounds(subres; idxs=idxs)   # your existing bounding logic
    old_ts = raw.ts                       # might be shorter or mismatched
    raw_sol = raw.sol                     # e.g., raw_sol[i] = (low_array, up_array)

    # We'll build a new dictionary with same keys but resampled arrays.
    new_sol = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    for i in idxs
        # (low_array, up_array)
        (lA, uA) = raw_sol[i]
        # Interpolate each point in new_ts
        lA_new = [linear_interp(t, old_ts, lA) for t in new_ts]
        uA_new = [linear_interp(t, old_ts, uA) for t in new_ts]
        new_sol[i] = (lA_new, uA_new)
    end

    # Return the same structure, but time points replaced with new_ts
    return SimulationResult(new_sol, raw.vars, new_ts, raw.kind)
end
