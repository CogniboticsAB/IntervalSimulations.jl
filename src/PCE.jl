# -------------------------------------------
# Legendre utility functions
# -------------------------------------------

function legendre_polynomials(ξ)
    return [
        1.0,
        ξ,
        0.5*(3ξ^2 - 1),
        0.5*(5ξ^3 - 3ξ),
        (1/8)*(35ξ^4 - 30ξ^2 + 3),
        (1/8)*(63ξ^5 - 70ξ^3 + 15ξ),
        (1/16)*(231ξ^6 - 315ξ^4 + 105ξ^2 - 5),
        (1/16)*(429ξ^7 - 693ξ^5 + 315ξ^3 - 35ξ),
        (1/128)*(6435ξ^8 - 12012ξ^6 + 6930ξ^4 - 1260ξ^2 + 35),
        (1/128)*(12155ξ^9 - 25740ξ^7 + 18018ξ^5 - 4620ξ^3 + 315ξ),
        (1/256)*(46189ξ^10 - 109395ξ^8 + 90090ξ^6 - 30030ξ^4 + 3465ξ^2 - 63)
    ]
end

function generate_total_degree_multi_indices(d, p)
    return [i for i in Iterators.product((0:p for _ in 1:d)...) if sum(i) <= p]
end

# -------------------------------------------
# Collocation setup
# -------------------------------------------

function generate_collocation_nodes(param_intervals::Dict, poly_order::Int)
    param_array = collect(param_intervals)
    d = length(param_array)
    N = poly_order + 1
    xi_1D, _ = FastGaussQuadrature.gausslegendre(N)

    scaled_nodes = [(xi_1D .+ 1) .* ((RA.sup(inter) - RA.inf(inter)) / 2) .+ RA.inf(inter) for (_, inter) in param_array]
    xi_nodes = fill(xi_1D, d)

    param_keys = first.(param_array)
    scaled_product = collect(product(scaled_nodes...))
    xi_product = collect(product(xi_nodes...))

    collocation_nodes = [Dict(param_keys[j] => sp[j] for j in 1:d) for sp in scaled_product]
    xi_combinations = [collect(xp) for xp in xi_product]
    return collocation_nodes, xi_combinations
end

function generate_quadrature_weights(d::Int, poly_order::Int)
    N = poly_order + 1
    _, weights_1D = FastGaussQuadrature.gausslegendre(N)
    weights_product = collect(product((weights_1D for _ in 1:d)...))
    return [prod(w) for w in weights_product]
end

# -------------------------------------------
# Main PCE routine
# -------------------------------------------
"""
    solveInterval(prob, tspan; var_dict=Dict(), dt=0.01, poly_order=2,
                  pce_solver=Rodas5(), interesting_variables=[], 
                  extra_callocation=0, verbose=false)

Performs Polynomial Chaos Expansion (PCE) using collocation on the ODESystem `prob`.
Returns a `SimulationResult` with metadata and time-aligned solutions.

Arguments:
- `prob`: The `ODESystem` to simulate.
- `tspan`: Tuple or vector with start and stop time.
- `var_dict`: Dictionary overriding default parameter intervals.
- `dt`: Save interval.
- `poly_order`: Number of Legendre basis polynomials.
- `pce_solver`: ODE solver used internally at each node.
- `interesting_variables`: Extra variables (e.g., outputs) to track.
- `extra_callocation`: Extra nodes for accuracy.
- `verbose`: Show print/debug output.

Returns:
- `SimulationResult` with `:type => :pce`.
"""
function solve_pce(prob, tspan; var_dict=Dict(), dt=0.01, poly_order=2,
                       pce_solver=Rodas5(), interesting_variables=[],
                       extra_callocation=0, verbose=false)

    # Step 1: Extract intervals
    param_intervals = getIntervals(prob, var_dict)

    # Step 2: Generate collocation points
    d = length(param_intervals)
    nodes, xi_nodes = generate_collocation_nodes(param_intervals, poly_order + extra_callocation)
    weights = generate_quadrature_weights(d, poly_order + extra_callocation)
    save_times = tspan[1]:dt:tspan[2]
    N = length(nodes)

    # Step 3: Determine variables to observe
    state_vars = MTK.unknowns(prob)
    all_vars = [state_vars...; interesting_variables...]

    # Step 4: Solve the system at each node
    solutions = Vector{MTK.ODESolution}(undef, N)
    for i in ProgressBars.ProgressBar(1:N)
        p_dict = copy(nodes[i])
        ic_dict = Dict(u => p_dict[u] for u in state_vars if haskey(p_dict, u))
        for u in keys(ic_dict)
            delete!(p_dict, u)
        end
        #prob1 = deepcopy(prob)
        odeprob = MTK.ODEProblem(prob, ic_dict, tspan, p_dict)
        solutions[i] = DifferentialEquations.solve(odeprob, pce_solver, saveat=save_times)
    end

    # Step 5: Sanity check
    for (i, sol) in enumerate(solutions)
        if sol.retcode != :Success
            error("Solver failed at node $i with retcode $(sol.retcode)")
        end
    end

    # Step 6: Return SimulationResult
    return SimulationResult(
        (
            sol = solutions,
            xi_nodes = xi_nodes,
            collocation_nodes = nodes,
            quadrature_weights = weights,
            save_times = save_times,
            d = d,
            poly_order = poly_order,
            extra_callocation = extra_callocation,
            sys = prob
        ),
        all_vars,
        save_times,
        Dict(
            :type => :pce,
            :dt => dt,
            :solver => pce_solver,
            :param_intervals => param_intervals,
            :number_of_callocation_nodes => N,
            :problem => prob
        )
    )
end

function calculate_bounds_pce(result::SimulationResult; idxs=nothing)
    sys = result.kind[:problem]
    data = result.sol
    d = data.d
    poly_order = data.poly_order
    xi_nodes = data.xi_nodes
    weights = data.quadrature_weights
    solutions = data.sol
    save_times = data.save_times

    all_vars = result.vars isa Tuple ? collect(result.vars) : result.vars
    num_times = length(save_times)

    # Handle variable selection
    idxlist =
        isnothing(idxs) ? all_vars :
        isa(idxs, Tuple) ? collect(idxs) :
        isa(idxs, AbstractVector) ? idxs :
        [idxs]

    selected_vars = [i isa Int ? all_vars[i] : i for i in idxlist]

    multi_indices = generate_total_degree_multi_indices(d, poly_order)
    M = length(multi_indices)
    N_nodes = length(solutions)

    # Polynomial basis evaluations
    phi_table = Matrix{Float64}(undef, N_nodes, M)
    @threads for k in 1:N_nodes
        ξs = xi_nodes[k]
        for m_idx in 1:M
            multi_idx = multi_indices[m_idx]
            phi_table[k, m_idx] = prod(legendre_polynomials(ξs[i])[multi_idx[i] + 1] for i in 1:d)
        end
    end

    norm_factors = [prod(((2 * i + 1) / 2) for i in mi) for mi in multi_indices]

    legendre_intervals = [
        IA.interval(1.0), IA.interval(-1.0, 1.0), IA.interval(-0.5, 1.0),
        IA.interval(-1.0, 1.0), IA.interval(-0.4286, 1.0), IA.interval(-1.0, 1.0),
        IA.interval(-0.4147, 1.0), IA.interval(-1.0, 1.0), IA.interval(-0.4097, 1.0),
        IA.interval(-1.0, 1.0), IA.interval(-0.4073, 1.0)
    ]
    poly_interval = [prod(legendre_intervals[i+1] for i in mi) for mi in multi_indices]

    zero_multi_idx = ntuple(_ -> 0, d)

    results = Dict{Any, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}()

    for unk in ProgressBars.ProgressBar(selected_vars)
        coeff_dict = Dict(mi => zeros(num_times) for mi in multi_indices)
        mean_vals = zeros(num_times)
        lower_vals = zeros(num_times)
        upper_vals = zeros(num_times)

        for t_idx in 1:num_times
            y_vals = [solutions[k][unk][t_idx] for k in 1:N_nodes]
            enclosure = IA.interval(0.0)

            @threads for m_idx in 1:M
                sum_val = sum(weights[k] * y_vals[k] * phi_table[k, m_idx] for k in 1:N_nodes)
                coeff_dict[multi_indices[m_idx]][t_idx] = norm_factors[m_idx] * sum_val
            end

            for m_idx in 1:M
                enclosure += coeff_dict[multi_indices[m_idx]][t_idx] * poly_interval[m_idx]
            end

            lower_vals[t_idx] = IA.inf(enclosure)
            upper_vals[t_idx] = IA.sup(enclosure)
            mean_vals[t_idx] = coeff_dict[zero_multi_idx][t_idx]
        end

        results[unk] = (mean_vals, lower_vals, upper_vals)
    end

    return SimulationResult(
        results,
        selected_vars,
        save_times,
        merge(result.kind, Dict(:type => :pce_bounded, :idxs => idxlist))
    )
end


function split_intervals(prob, var_splits::Dict)
    # Start with one base interval set
    base = getIntervals(prob, Dict())
    inters = [copy(base)]

    for var in keys(var_splits)
        n = var_splits[var]
        new_inters = []

        for d in inters
            iv = d[var]
            lb = ReachabilityAnalysis.inf(iv)
            ub = ReachabilityAnalysis.sup(iv)
            Δ = (ub - lb) / n

            for i in 0:(n-1)
                d_new = copy(d)
                d_new[var] = IA.Interval((lb + i*Δ),(lb + (i+1)*Δ))
                push!(new_inters, d_new)
            end
        end

        inters = new_inters
    end

    return inters
end


# This function runs the PCE simulation over split subintervals.
# split_counts is a dictionary like Dict(:l1 => 2), meaning the base interval for l1
# is divided equally into 2 subintervals.
function solve_pce_split(prob, tspan;
    var_dict=Dict(),
    dt=0.01,
    poly_order=2,
    pce_solver=Rodas5(),
    interesting_variables=[],
    extra_callocation=0,
    verbose=false,
    split_counts=Dict())  # e.g. Dict(:l1 => 2)

    # If no splitting, just delegate to solve_pce directly
    if isempty(split_counts)
        return solve_pce(prob, tspan;
                         var_dict=var_dict,
                         dt=dt,
                         poly_order=poly_order,
                         pce_solver=pce_solver,
                         interesting_variables=interesting_variables,
                         extra_callocation=extra_callocation,
                         verbose=verbose)
    end

    # Get split interval dicts using your utility
    split_var_dicts = split_intervals(prob, split_counts)

    sim_results = SimulationResult[]
    for new_var_dict in split_var_dicts
        # Add any overrides from user-specified var_dict (but keep split intervals)
        for (k, v) in var_dict
            new_var_dict[k] = v
        end
        
        # Convert any LazySets or odd intervals
        #new_var_dict = Dict(k => to_IA_interval(v) for (k, v) in new_var_dict)

        # Run the simulation for this subinterval configuration
        sim_res = solve_pce(prob, tspan;
                            var_dict=new_var_dict,
                            dt=dt,
                            poly_order=poly_order,
                            pce_solver=pce_solver,
                            interesting_variables=interesting_variables,
                            extra_callocation=extra_callocation,
                            verbose=verbose)
        push!(sim_results, sim_res)
    end

    # Assume common ts/vars
    common_ts = sim_results[1].ts
    common_vars = sim_results[1].vars

    return SimulationResult(sim_results, common_vars, common_ts,
                            Dict(:type => :split_pce, :split_counts => split_counts))
end

# This function takes a SimulationResult from solve_pce_split (i.e. with kind :split_pce)
# and computes overall bounds by aggregating the individual bounds from each simulation.
# It assumes that calculate_bounds_pce(sim_res) returns a SimulationResult in which,
# for each variable v, bounds are stored in a tuple (mean_vals, lower_vals, upper_vals).
function calculate_bounds_split(split_res::SimulationResult)
if split_res.kind[:type] != :split_pce
    error("Provided SimulationResult is not a split simulation result.")
end

# Retrieve the vector of simulation results.
sim_results = split_res.sol
# Compute bounds for each split-case simulation.
bounds_list = [calculate_bounds_pce(r) for r in sim_results]

common_ts = split_res.ts
common_vars = split_res.vars
final_bounds = Dict()

# For each variable, combine the lower and upper bounds time point–by–time point.
for v in common_vars
    lowers = [b.sol[v][2] for b in bounds_list]
    uppers = [b.sol[v][3] for b in bounds_list]
    num_times = length(common_ts)
    combined_lower = [minimum(getindex.(lowers, t)) for t in 1:num_times]
    combined_upper = [maximum(getindex.(uppers, t)) for t in 1:num_times]
    final_bounds[v] = (fill(NaN, num_times), combined_lower, combined_upper)
end

# Return a new SimulationResult with the overall bounds.
return SimulationResult(final_bounds, common_vars, common_ts,
                        Dict(:type => :pce_bounded_split, :origin => :split))
end