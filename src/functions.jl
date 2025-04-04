############################
# Example "functions.jl"
############################

include("PCElegendre.jl")

##############################################################################
# DATA STRUCTS
##############################################################################

struct IntervalSolution
    sol  # either a ReachSolution or the dict-of-means/lower/upper from Legendre
    vars # a list of variable names/symbols
    ts   # time vector, if applicable
    kind # either :reach or :pce or :bounded_reach
end

"""
    struct ScanningSolutions

Container for multiple ODESolutions from either scanning or Monte Carlo.

Fields:
- `solutions` : Vector of ODESolution
- `varnames`  : Tuple of variable symbols
- `meta`      : Arbitrary Dict with metadata (e.g. problem, tspan, grid_size)
- `kind`      : Symbol, e.g. `:scanning` or `:monte`
"""
struct ScanningSolutions
    solutions::Vector{ODESolution}
    varnames::Tuple{Vararg{Num}}
    meta
    kind::Symbol
end

##############################################################################
# UTILITY: getIntervals
##############################################################################

function getIntervals(fol, pval)
    d = defaults(fol)

    new_u = Dict(kv for kv in d if kv.second isa IntervalArithmetic.Interval)

    # Convert keys in new_u to Num type
    converted_new_u = Dict(Num(k) => v for (k, v) in new_u)
    intervals = merge(converted_new_u, pval)
    final_intervals = Dict(Num(k) => v for (k, v) in intervals)
    return final_intervals
end

##############################################################################
# solveInterval
##############################################################################

function solveInterval(prob, tspan; var_dict=Dict(), dt=0.01, poly_order=2, solver=TMJets21b(),
                       intresting_variables=[], print=false, pce_solver=Rodas5(),
                       extra_callocation=0, get_bounds=false)
    if prob isa ODESystem
        println("Legendre")
        # Get parameter intervals from the problem and provided variable values.
        param_intervals = getIntervals(prob, var_dict)
        # Instead of calling run_pce_interval_analysis directly, we call the new solve_pce routine,
        # which returns a data tuple containing all the precomputed info.
        data = solve_pce(prob, poly_order, tspan, dt, param_intervals, print, pce_solver;
                         extra_callocation=extra_callocation)
        
        ts = data.save_times
        num_nodes = poly_order + extra_callocation

        # Build a variable list from the unknowns and the extra interesting variables.
        vars = Tuple([unknowns(prob)..., intresting_variables...])
        return IntervalSolution(data, vars, ts,
            Dict(:type => :pce,
                 :poly_order => poly_order,
                 :dt => dt,
                 :param_intervals => param_intervals,
                 :solver => pce_solver,
                 :intresting_variables => intresting_variables,
                 :number_of_callocation_nodes => num_nodes,
                 :problem => prob))

    elseif (prob isa Tuple && prob[1] isa InitialValueProblem && prob[3] == 2)
        println("IVP / Reachability 2")
        ra_sol = ReachabilityAnalysis.solve(prob[1], tspan=(tspan[1], tspan[2]), solver)
        save_times = tspan[1]:dt:tspan[2]
        vars = prob[2]
        return IntervalSolution(
            ra_sol,
            vars,
            save_times,
            Dict(:type => :reach, :solver => solver, :problem => prob[1], :dt => dt)
        )
    else
        error("solveInterval: unrecognized problem type")
    end
end

"""
    getBounds(ra_sol; indices=1:length(ra_sol.vars))

After running `solveInterval(...)` with a :reach solution, this further “bounds”
the solution using `TaylorModels.quadratic_fast_bounder`. The returned 
`IntervalSolution` will have `kind[:type] == :bounded_reach`.
"""
function getBounds(ra_sol; indices=1:length(ra_sol.vars))
    sols = [
        [
            TaylorModels.quadratic_fast_bounder(
                ReachabilityAnalysis.evaluate(ra_sol.sol.F.Xk[x], ra_sol.sol.F.Xk[x].Δt)[k]
            )
            for x in 1:length(ra_sol.sol)
        ]
        for k in indices
    ]
    save_times = [
        0.5*(sup(ra_sol.sol.F.Xk[x].Δt[1]) + ReachabilityAnalysis.inf(ra_sol.sol.F.Xk[x].Δt[1]))
        for x in 1:length(ra_sol.sol)
    ]
    ra_sol.kind[:type] = :bounded_reach
    return IntervalSolution(sols, ra_sol.vars, save_times, ra_sol.kind)
end

##############################################################################
# PLOTTING IntervalSolution
##############################################################################

function intervalPlot(intervalsol::IntervalSolution; var_names=["y1","y2"], idxs=(0,1), kwargs...)
    fig = plot()
    intervalPlot!(fig, intervalsol; var_names=var_names, idxs=idxs, kwargs...)
    return fig
end

function intervalPlot!(fig, intervalsol::IntervalSolution; var_names=["y1","y2"], idxs=(0,1), kwargs...)
    if intervalsol.kind[:type] == :reach
        @info "Plotting ReachabilityAnalysis solution"
        plot!(
            fig,
            intervalsol.sol;
            vars=idxs,
            linealpha=0.0,
            fillalpha=0.3,
            label=false,
            kwargs...
        )

    elseif intervalsol.kind[:type] == :pce
        @info "Plotting Legendre/PCE solution"
        _plotPCE!(fig, intervalsol, idxs; var_names=var_names, kwargs...)

    elseif intervalsol.kind[:type] == :bounded_reach
        @info "Plotting bounded reach solution"
        i1, i2 = idxs
        local label_suffix = "??"
        # If you want: label_suffix = var_names[1] if it exists, etc.

        if i1 == 0
            # time vs i2
            plot!(
                intervalsol.ts,
                ReachabilityAnalysis.sup.(intervalsol.sol[i2]);
                color=:blue, label="Bounds, TM", xlabel="time (s)", ylabel="$(var_names[1])",
                kwargs...
            )
            plot!(
                intervalsol.ts,
                ReachabilityAnalysis.inf.(intervalsol.sol[i2]);
                color=:blue, label="",
                kwargs...
            )
        else
            # phase
            plot!(
                ReachabilityAnalysis.sup.(intervalsol.sol[i1]),
                ReachabilityAnalysis.sup.(intervalsol.sol[i2]);
                color=:blue, label="",
                xlabel="h", title="$(var_names[1]) vs $(var_names[2])",
                kwargs...
            )
            plot!(
                ReachabilityAnalysis.inf.(intervalsol.sol[i1]),
                ReachabilityAnalysis.inf.(intervalsol.sol[i2]);
                color=:blue, label="",
                kwargs...
            )
        end
    else
        @warn "intervalPlot!: Unrecognized :type => $(intervalsol.kind[:type])"
    end
    return fig
end

# Helper to handle PCE
function _plotPCE!(fig, intervalsol::IntervalSolution, idxs; var_names=["y1","y2"], kwargs...)
    sol_dict = intervalsol.sol  # e.g. sol_dict[var_symbol] => (m, l, u)
    t_vec    = intervalsol.ts
    vars     = intervalsol.vars

    # Distinguish shapes of idxs
    if isa(idxs, Tuple)
        if length(idxs) == 2
            i1, i2 = idxs
            if i1 == 0
                var_symbol = vars[i2]
                mvec, lvec, uvec = sol_dict[var_symbol]
                plot!(
                    fig,
                    t_vec,
                    lvec;
                    ls=:dash,
                    color=:red,
                    label="Bound PCE",
                    xlabel="time (s)",
                    ylabel="$(var_symbol)",
                    kwargs...
                )
                plot!(fig, t_vec, uvec; ls=:dash, color=:red, label="", kwargs...)
            else
                # phase plot, means
                var_sym1 = vars[i1]
                var_sym2 = vars[i2]
                m1, l1, u1 = sol_dict[var_sym1]
                m2, l2, u2 = sol_dict[var_sym2]

                x_polygon = [u1...; reverse(l1)...; u1[1]]
                y_polygon = [u2...; reverse(l2)...; u2[1]]

                plot!(
                    fig,
                    x_polygon,
                    y_polygon;
                    seriestype=:shape,
                    linecolor=:transparent,
                    label=false,
                    kwargs...
                )
                plot!(fig, m1, m2; color=:black, label="mean($var_sym2 vs $var_sym1)", kwargs...)
                plot!(fig, l1, l2; color=:black, label="low($var_sym2)", kwargs...)
                plot!(fig, u1, u2; color=:black, label="high($var_sym2)", kwargs...)
            end
        else
            @warn "idxs tuple length ≠ 2 not supported in PCE. Doing nothing."
        end

    elseif isa(idxs, AbstractVector) || isa(idxs, UnitRange)
        # subplots
        for i in idxs
            var_symbol = vars[i]
            mvec, lvec, uvec = sol_dict[var_symbol]
            plot!(
                fig,
                t_vec,
                mvec;
                ribbon = (uvec .- mvec, mvec .- lvec),
                label  = "mean ± bounds $(var_symbol)",
                fillalpha=0.2,
                kwargs...
            )
        end
    else
        @warn "idxs not recognized for PCE. Please pass (0,i), (i,j), or a range."
    end
    return fig
end

##############################################################################
# SCANNING + MONTECARLO
##############################################################################

"""
    solutions = solveScanning(sys, tspan, grid_size; var_dict, dt, intresting_variables)

Performs a parameter scan over `var_dict` (e.g. Dict(:p => 1 ± 0.1)) 
using `grid_size` points per parameter dimension. 
Returns a `ScanningSolutions(kind=:scanning)`.
"""
function solveScanning(sys, tspan, grid_size; var_dict=Dict(), dt=0.01, intresting_variables=[])
    param_ranges = getIntervals(sys, var_dict)
    fine_grid_points = grid_size

    t_start, t_end = tspan
    timesteps = t_start:dt:t_end

    # Build ranges
    ranges_array = Vector{StepRangeLen{Float64}}()
    for (sym, inter) in param_ranges
        lo, hi = inf(inter), sup(inter)
        push!(ranges_array, range(lo, hi, length=fine_grid_points))
    end

    param_grid = collect(product(ranges_array...))

    solutions = Vector{ODESolution}(undef, length(param_grid))
    total = length(param_grid)

    for (i, combo) in enumerate(param_grid)
        param_dict = Dict()
        local_j = 1
        for (sym, _) in param_ranges
            param_dict[sym] = combo[local_j]
            local_j += 1
        end

        prob = ODEProblem(sys, nothing, (t_start, t_end), param_dict)
        sol = solve(prob, saveat=timesteps)
        solutions[i] = sol

        println("Progress: ", round((i/total*100), digits=2), "%")
    end

    labels = Tuple([unknowns(sys)..., intresting_variables...])
    return ScanningSolutions(solutions, labels, Dict(:problem=>sys, :ts=>tspan, :grid_size=>grid_size), :scanning)
end

"""
    solutions = solveMonteCarlo(sys, tspan, num_samples; var_dict=Dict(), dt=0.01, interesting_variables=[])

Performs a Monte-Carlo parameter sampling over `var_dict` 
with `num_samples` random draws. Returns `ScanningSolutions(kind=:monte)`.
"""
function solveMonteCarlo(sys, tspan, num_samples; var_dict=Dict(), dt=0.01, interesting_variables=[])
    param_ranges = getIntervals(sys, var_dict)
    N = num_samples

    t_start, t_end = tspan
    timesteps = t_start:dt:t_end

    solutions = Vector{ODESolution}(undef, N)

    for i in 1:N
        param_dict = Dict()
        for (sym, inter) in param_ranges
            lo, hi = inf(inter), sup(inter)
            param_dict[sym] = rand() * (hi - lo) + lo
        end

        prob = ODEProblem(sys, nothing, (t_start, t_end), param_dict)
        sol = solve(prob, saveat=timesteps)
        solutions[i] = sol

        println("Monte-Carlo sample ", i, " of ", N, ": done.")
    end

    labels = Tuple([unknowns(sys)..., interesting_variables...])
    return ScanningSolutions(
        solutions,
        labels,
        Dict(:problem=>sys, :ts=>tspan, :num_samples=>num_samples),
        :monte
    )
end

##############################################################################
# BOUNDS + PLOT FOR ScanningSolutions
##############################################################################

"""
    computeSolutionBounds(solutions::ScanningSolutions; idxs=(0,1))

Compute min/max arrays over all solutions in `solutions.solutions`. 
Supports `idxs` as an Int or a tuple (0,i), etc. 
Returns a dictionary with `:times, :max1, :min1, ...`.
Also includes `:kind => solutions.kind` to identify scanning vs. MC.
"""
function computeSolutionBounds(solutions::ScanningSolutions; idxs=(0,1))
    if isempty(solutions.solutions)
        error("No solutions in scanning/monte object.")
    end

    println("Compute bounds for ", solutions.kind == :scanning ? "SM" : "MC")

    times = solutions.solutions[1].t
    n = length(times)

    # We'll do a simple approach for either idxs is an Int or a tuple (0,i).
    # You can expand logic for other cases. Just as an example:
    max1 = fill(-Inf, n)
    min1 = fill( Inf, n)
    max2 = fill(-Inf, n)
    min2 = fill( Inf, n)

    if isa(idxs, Int)
        var_sym = solutions.varnames[idxs]
        for j in 1:n
            vals = [sol[var_sym][j] for sol in solutions.solutions]
            max1[j] = maximum(vals)
            min1[j] = minimum(vals)
        end
    elseif isa(idxs, Tuple{Int, Int})
        i1, i2 = idxs
        # If i1=0 => time vs i2
        # If not, do 2D bounding
        if i1 == 0
            var_sym = solutions.varnames[i2]
            for j in 1:n
                vals = [sol[var_sym][j] for sol in solutions.solutions]
                max1[j] = maximum(vals)
                min1[j] = minimum(vals)
            end
        else
            var1 = solutions.varnames[i1]
            var2 = solutions.varnames[i2]
            for j in 1:n
                vals1 = [sol[var1][j] for sol in solutions.solutions]
                vals2 = [sol[var2][j] for sol in solutions.solutions]
                max1[j] = maximum(vals1)
                min1[j] = minimum(vals1)
                max2[j] = maximum(vals2)
                min2[j] = minimum(vals2)
            end
        end
    else
        @warn "computeSolutionBounds doesn't handle idxs=$idxs fully."
    end

    return Dict(
        :kind => solutions.kind,    # scanning or monte
        :times => times,
        :max1  => max1,
        :min1  => min1,
        :max2  => max2,
        :min2  => min2,
        :idxs  => idxs
    )
end

"""
    plotSolutionBounds(solutions, bounds; kwargs...)

Creates a new plot from the precomputed bounding arrays in `bounds`.
Calls `plotSolutionBounds!`.
"""
function plotSolutionBounds(solutions::ScanningSolutions, bounds::Dict{Symbol,<:Any}; kwargs...)
    plt = plot()
    plotSolutionBounds!(plt, solutions, bounds; kwargs...)
    return plt
end

"""
    plotSolutionBounds!(plt, solutions, bounds; kwargs...)

Plot *in place* the bounding arrays from `computeSolutionBounds`.
It will label “Bound SM” or “Bound MC” depending on `solutions.kind`.
"""
function plotSolutionBounds!(
    plt,
    solutions::ScanningSolutions,
    bounds::Dict{Symbol,<:Any};
    color=:blue,
    alpha=0.5,
    label="",
    kwargs...
)
    # Retrieve arrays
    times = bounds[:times]
    max1  = bounds[:max1]
    min1  = bounds[:min1]
    max2  = bounds[:max2]
    min2  = bounds[:min2]
    idxs  = bounds[:idxs]

    # Decide label suffix
    label_suffix = solutions.kind == :scanning ? "SM" :
                   solutions.kind == :monte    ? "MC" : "??"

    # Simple logic
    if isa(idxs, Int)
        # single variable
        plot!(plt, times, max1; label="Bound $label_suffix", color=color, alpha=alpha, kwargs...)
        plot!(plt, times, min1; label="", color=color, alpha=alpha, kwargs...)
    elseif isa(idxs, Tuple{Int,Int})
        i1, i2 = idxs
        if i1 == 0
            # time vs i2
            plot!(plt, times, max1; label="Bound $label_suffix", color=color, alpha=alpha, kwargs...)
            plot!(plt, times, min1; label="", color=color, alpha=alpha, kwargs...)
        else
            # 2D bounding
            plot!(plt, min1, min2; label="BoundLow $label_suffix", color=color, alpha=alpha, kwargs...)
            plot!(plt, max1, max2; label="BoundHigh $label_suffix", color=color, alpha=alpha, kwargs...)
        end
    else
        @warn "plotSolutionBounds!: Not fully handling idxs=$idxs."
    end

    return plt
end

"""
    getUnifiedBounds(sol; kwargs...)

Compute and return the bounds for a given solution object `sol`
regardless of its type.

- For a PCE solution (`sol.kind[:type] == :pce`), the bounds are assumed
  to be already computed and stored in `sol.sol` (or you may call your
  PCE bounds routine here).
- For a reachability solution (`:reach`), the function calls the existing
  `getBounds` routine to generate a bounded reach solution.
- If `sol.kind[:type] == :bounded_reach`, the solution is already bounded.
- For scanning or Monte Carlo solutions (`:scanning` or `:monte`), the
  function uses `computeSolutionBounds` to calculate the bounds.
"""
function getUnifiedBounds(sol; kwargs...)
    sol_type = sol.kind[:type]
    if sol_type == :pce
        @info "PCE solution detected. Computing bounds using calculate_bounds_pce."
        data = calculate_bounds_pce(sol, sol.kind[:intresting_variables])
        return IntervalSolution(data[1], sol.vars, sol.ts,sol.kind)
    elseif sol_type == :reach
        @info "Reach solution detected. Computing bounds using TaylorModels."
        # Uses your getBounds function (which returns an IntervalSolution with kind :bounded_reach)
        return getBounds(sol; kwargs...)
    elseif sol_type == :bounded_reach
        @info "Bounded reach solution detected. Returning as-is."
        return sol
    elseif sol_type in (:scanning, :monte)
        @info "Scanning/Monte Carlo solution detected. Computing bounds via computeSolutionBounds."
        # Compute and return a bounds dictionary from scanning/Monte solutions
        return computeSolutionBounds(sol; kwargs...)
    else
        error("getUnifiedBounds: Unrecognized solution type: $(sol_type)")
    end
end