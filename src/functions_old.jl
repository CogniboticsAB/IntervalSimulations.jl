include("PCElegendre.jl")

struct IntervalSolution
    sol  # either a ReachSolution or the dict-of-means/lower/upper from Legendre
    vars # a list of variable names/symbols
    ts   # time vector, if applicable
    kind # either :reach or :pce
end

struct ScanningSolutions
    solutions::Vector{ODESolution}
    varnames::Tuple{Vararg{Num}}
    meta
    kind
end

function getIntervals(fol,pval)
    d = defaults(fol)

    new_u= Dict(kv for kv in d if (kv.second isa IntervalArithmetic.Interval))

    # Convert keys in new_u to Num type
    converted_new_u = Dict(Num(k) => v for (k, v) in new_u)
    intervals = merge(converted_new_u, pval)
    final_intervals = Dict(Num(k) => v for (k, v) in intervals)
    #println(final_intervals)
    return final_intervals
end
##############################################################################
# INTERVAL
##############################################################################

function solveInterval(prob, tspan; var_dict=Dict(), dt=0.01, poly_order=2, solver=TMJets21b(), intresting_variables = [], print = false, pce_solver = Rodas5(), extra_callocation = 0, get_bounds = true)
    if prob isa ODESystem
        println("Legendre")
        # Suppose runlegendre returns (solutionDict, timeVector), or something similar
        param_intervals = getIntervals(prob,var_dict)
        pce_sol, ts, number_of_callocation_nodes =  run_pce_interval_analysis(prob, poly_order, tspan, dt, param_intervals, intresting_variables, print, pce_solver, extra_callocation=extra_callocation)
        #pce_sol, ts =  runlegendre(prob, dim, tspan, dt, getIntervals(prob,var_dict), intresting_variables, print)
        #return pce_sol
        # You can store the unknowns(prob) in a variables list if you want
        vars = Tuple([unknowns(prob)...,intresting_variables...])
        return IntervalSolution(pce_sol, vars, ts, Dict(:type=>:pce, :poly_order => poly_order, :dt=>dt,:param_intervals =>param_intervals,:solver => pce_solver, :intresting_variables=> intresting_variables,:number_of_callocation_nodes=>number_of_callocation_nodes, :problem =>prob))
    elseif (prob isa Tuple && prob[1] isa InitialValueProblem && prob[3]==2)
        println("IVP / Reachability 2")
        ra_sol = ReachabilityAnalysis.solve(prob[1], tspan=(tspan[1], tspan[2]), solver)
        save_times = tspan[1]:dt:tspan[2]
        #sols = [[TaylorModels.quadratic_fast_bounder(ReachabilityAnalysis.evaluate(ra_sol.F.Xk[x], ra_sol.F.Xk[x].Δt)[k]) for x in 1:length(ra_sol)] for k in 1:8]
        #save_times = [0.5*(sup(ra_sol.F.Xk[x].Δt[1])+ ReachabilityAnalysis.inf(ra_sol.F.Xk[x].Δt[1])) for x in 1:length(ra_sol)]
        # prob[2] might be the symbol list or variable set you want
        vars = prob[2]
        return IntervalSolution(ra_sol, vars, save_times, Dict(:type=>:reach,:solver => solver, :problem => prob[1], :dt => dt))
    else
        error("solveInterval: unrecognized problem type")
    end
end
function getBounds(ra_sol; indices = 1:length(ra_sol.vars))
    sols = [[TaylorModels.quadratic_fast_bounder(ReachabilityAnalysis.evaluate(ra_sol.sol.F.Xk[x], ra_sol.sol.F.Xk[x].Δt)[k]) for x in 1:length(ra_sol.sol)] for k in indices]
    save_times = [0.5*(sup(ra_sol.sol.F.Xk[x].Δt[1])+ ReachabilityAnalysis.inf(ra_sol.sol.F.Xk[x].Δt[1])) for x in 1:length(ra_sol.sol)]
    ra_sol.kind[:type]=:bounded_reach
    return IntervalSolution(sols, ra_sol.vars, save_times, ra_sol.kind)
end

function intervalPlot(intervalsol::IntervalSolution; var_names=["y1","y2"], idxs=(0,1), kwargs...)
    fig = plot()
    intervalPlot!(fig, intervalsol; var_names=var_names, idxs=idxs, kwargs...)
    return fig
end

function intervalPlot!(fig, intervalsol::IntervalSolution; var_names=["y1","y2"], idxs=(0,1), kwargs...)
    if intervalsol.kind[:type] == :reach
        @info "Plotting ReachabilityAnalysis solution"
        # Since `intervalsol.sol` is a ReachSolution, we can do
        # plot!(..., idxs=...) just like ODESolution’s recipe, but it’s slightly different:
        #   - Typically in RA, the first coordinate (0) is time.
        #   - So if `idxs=(0,2)`, that means time vs. x2(t).
        #   - If `idxs=(1,2)`, that is x2 vs x1.

        plot!(
            fig,
            intervalsol.sol;
            vars=idxs,
            linealpha=0.0,   # or lw=0
            fillalpha=0.3,
            label=false
        )

    elseif intervalsol.kind[:type] == :pce
        @info "Plotting Legendre/PCE solution"
        # This is a dictionary-of-mean/lower/upper for each variable => (m, l, u).
        # We need to interpret `idxs`. If the user gave us:
        #   - a single tuple (0, i) => time vs. var i
        #   - a range 1:3 => subplots for var 1,2,3 vs. time
        #   - a tuple (i, j) with i != 0 => phase plot of var j vs var i (means)
        #
        # For simplicity, let's implement only two “modes”:
        #   (0, i) or a range of ints. If the user tries (i, j) with i!=0,
        #   we do a “phase plot” of the means only. (No intervals.)
        #
        # We'll do a small helper below.

        _plotPCE!(fig, intervalsol, idxs; var_names=var_names)
    elseif intervalsol.kind[:type] == :bounded_reach
        i1, i2 = idxs
        if i1 == 0
            plot!(intervalsol.ts, ReachabilityAnalysis.sup.(intervalsol.sol[i2]);color = :blue, label = "Bounds, TM", xlabel="$(LaTeXStrings.L"time") (s)", ylabel ="$(var_names[1])", kwargs...)
            plot!(intervalsol.ts, ReachabilityAnalysis.inf.(intervalsol.sol[i2]);color=:blue, label = "", kwargs...)
        else
            plot!(ReachabilityAnalysis.sup.(intervalsol.sol[i1]), ReachabilityAnalysis.sup.(intervalsol.sol[i2]);color = :blue, label = "",xlabel="h", title = "$(var_names[1]) vs $(var_names[2])", kwargs...)
            plot!(ReachabilityAnalysis.inf.(intervalsol.sol[i1]), ReachabilityAnalysis.inf.(intervalsol.sol[i2]);color = :blue, label = "", kwargs...)
            
        end
    end
    return fig
end

# Helper to handle (0, i) or (i, j) or 1:3, etc.
function _plotPCE!(fig, intervalsol::IntervalSolution, idxs; var_names=["y1","y2"])
    sol_dict = intervalsol.sol  # e.g. sol_dict[var_symbol] => (m, l, u)
    t_vec    = intervalsol.ts
    vars     = intervalsol.vars


    # Distinguish the different shapes of idxs
    if isa(idxs, Tuple)
        if length(idxs) == 2
            i1, i2 = idxs
            # if i1 == 0, then it's "time vs. state i2"
            if i1 == 0
                var_symbol = vars[i2]  # e.g. the second variable

                mvec, lvec, uvec = sol_dict[var_symbol]

                #plot!(
                #    t_vec, lvec,
                    #fillrange  = uvec,   # Fill from lvec up to uvec
                    #fillalpha  = 0.1,
                    #fillcolor  = :red,
                #    linecolor  = :blue,
                #    linewidth  = 0,      # <-- No visible line for lvec
                #    label      = false,
                #    xlabel="$(LaTeXStrings.L"time") (s)"
                #)
                #plot!(t_vec, mvec, label = "$(var_symbol)", linewidth = 2, color = :black)
                plot!(fig, t_vec, lvec, label="Bound PCE", ls=:dash, color = :red, xlabel="$(LaTeXStrings.L"time") (s)", ylabel = "$(var_symbol)")
                plot!(fig, t_vec, uvec, label="", ls=:dash, color =:red)
            else
                # "phase plot" of var i2 vs var i1 (only the means)
                var_sym1 = vars[i1]
                var_sym2 = vars[i2]
                m1, l1, u1 = sol_dict[var_sym1]
                m2, l2, u2 = sol_dict[var_sym2]

                x_polygon = [u1...; reverse(l1)...; u1[1]]
                y_polygon = [u2...; reverse(l2)...; u2[1]]

                plot!(
                    x_polygon, y_polygon,
                    seriestype = :shape,
                    #fillalpha  = 0.0,
                    #fillcolor  = :red,
                    linecolor  = :transparent,  # Hide the polygon’s boundary
                    label      = false
                )

                # Optionally plot the mean trajectory on top
                plot!(m1, m2, color=:black, label="$(var_sym2) vs $(var_sym1)")
                
                #plot!(fig, m1, m2, label="mean $(var_sym2) vs $(var_sym1)")
                plot!(fig, l1, l2, color=:black,label="low $(var_sym2) vs $(var_sym1)")
                plot!(fig, u1, u2, color=:black,label="high $(var_sym2) vs $(var_sym1)")
            end
        else
            @warn "idxs as a tuple of length ≠ 2 not supported in PCE. Doing nothing."
        end

    elseif isa(idxs, AbstractVector) || isa(idxs, UnitRange)
        # e.g. idxs=1:3 => subplots for each variable (time vs that var)
        # We can do them all on the same subplot or do subplots. If you want subplots,
        # you can do something like `plot!(..., layout=(N,1))`, but let's keep it simpler:
        for i in idxs
            var_symbol = vars[i]
            mvec, lvec, uvec = sol_dict[var_symbol]
            #plot!(fig, t_vec, mvec, label="mean $(var_symbol)")
            #plot!(fig, t_vec, lvec, label="low $(var_symbol)", ls=:dash)
            #plot!(fig, t_vec, uvec, label="high $(var_symbol)", ls=:dash)
            plot!(
                fig,
                t_vec,
                mvec,
                ribbon = (uvec .- mvec, mvec .- lvec),
                label  = "mean ± bounds $(var_symbol)",
                fillalpha = 0.2,  # optional transparency for the band
            )
        end
    else
        @warn "idxs not recognized for PCE. Please pass (0,i), (i,j), or a range."
    end
end


##############################################################################
# SCANNING
##############################################################################

"""
    solutions = solveScanning(sys, tspan, grid_size, var_dict)

Performs a parameter scan over `var_dict` (e.g. Dict(:p => 1±0.1)) using `grid_size`
points per parameter dimension. Returns an array of `ODESolution`.
"""
function solveScanning(sys, tspan, grid_size; var_dict=Dict(), dt = 0.01, intresting_variables = [])
    # 1) Parameter intervals as Interval; e.g. var_dict[sym] => 1 ± 0.1
    #    We'll build a range(...) out of each.
    param_ranges = getIntervals(sys, var_dict)
    fine_grid_points = grid_size

    t_start, t_end = tspan
    timesteps = t_start:dt:t_end

    # Build a StepRangeLen for each parameter
    ranges_array = Vector{StepRangeLen{Float64}}()
    for (sym, inter) in param_ranges
        lo, hi = inf(inter), sup(inter)
        push!(ranges_array, range(lo, hi, length=fine_grid_points))
    end

    # Cartesian product over all parameter grids
    param_grid = collect(product(ranges_array...))

    # Solve each combination
    solutions = Vector{ODESolution}(undef, length(param_grid))
    total = length(param_grid)

    for (i, combo) in enumerate(param_grid)
        param_dict = Dict()
        local_j = 1
        for (sym, _) in param_ranges
            param_dict[sym] = combo[local_j]
            local_j += 1
        end

        # ODEProblem
        prob = ODEProblem(sys, nothing, (t_start, t_end), param_dict)
        sol = solve(prob, saveat=timesteps)
        solutions[i] = sol

        println("Progress: ", round((i/total*100), digits=2), " %")
    end
    labels = Tuple([unknowns(sys)..., intresting_variables...])
    return ScanningSolutions(solutions, labels, Dict(:problem =>sys,:ts=>tspan, :grid_size =>grid_size), :scanning)
end




"""
    plotScanning(solutions; idxs=(0,1), color=:gray, alpha=0.5, label=false)

Create a *new* plot for an array of ODESolutions, each solution plotted with the same `idxs=(...)`.
Some typical usages of `idxs` in DiffEqRecipes are:
  - `idxs=(0, 1)` => time vs. the 1st state
  - `idxs=(1, 2)` => phase plot of state 2 vs. state 1
  - `idxs=1:3`    => subplots for states 1,2,3 vs. time
"""
function plotScanning(
    solutions::IntervalSimulations.ScanningSolutions;
    idxs=(0,1),
    color=:gray,
    alpha=0.5
)
    plt = plot()  # create a new empty plot
    plotScanning!(plt, solutions; idxs=idxs, color=color, alpha=alpha)
    return plt
end

"""
    plotScanning!(plt, solutions; idxs=(0,1), color=:gray, alpha=0.5, label=false)

Plot *in place* into an existing plot `plt` for an array of ODESolutions, 
each solution plotted with the same `idxs=(...)`.
"""
function plotScanning!(
    plt,
    scanning_solutions::IntervalSimulations.ScanningSolutions;
    idxs=(0,1),
    color=:gray,
    alpha=0.5
)
    label_str=""
    times = scanning_solutions.solutions[1].t
    max1 = zeros(length(times))
    min1 = zeros(length(times))
    max2 = zeros(length(times))
    min2 = zeros(length(times))
    
    if idxs isa Num
        label_str = idxs
        for (j, time) in enumerate(times)
            max1[j] = maximum(scanning_solutions.solutions[i][label_str][j] for i in 1:length(scanning_solutions.solutions))
            min1[j] = minimum(scanning_solutions.solutions[i][label_str][j] for i in 1:length(scanning_solutions.solutions))
        end
        
        plot!(plt,times,max1, label = "Bounds$(label_str)")
        plot!(plt,times,min1,  label = "Minimum of $(label_str)")
    elseif idxs isa Tuple{Int64, Int64}
        if idxs[1] == 0
            arg1 = "t"
            arg2 = scanning_solutions.varnames[idxs[2]]
            for (j, time) in enumerate(times)
                max1[j] = maximum(scanning_solutions.solutions[i][arg2][j] for i in 1:length(scanning_solutions.solutions))
                min1[j] = minimum(scanning_solutions.solutions[i][arg2][j] for i in 1:length(scanning_solutions.solutions))
            end
            plot!(plt,times,max1)
            plot!(plt,times,min1)
        else
            arg1 = scanning_solutions.varnames[idxs[1]]
            arg2 = scanning_solutions.varnames[idxs[2]]
            for (j, time) in enumerate(times)
                max1[j] = maximum(scanning_solutions.solutions[i][arg1][j] for i in 1:length(scanning_solutions.solutions))
                min1[j] = minimum(scanning_solutions.solutions[i][arg1][j] for i in 1:length(scanning_solutions.solutions))
                max2[j] = maximum(scanning_solutions.solutions[i][arg2][j] for i in 1:length(scanning_solutions.solutions))
                min2[j] = minimum(scanning_solutions.solutions[i][arg2][j] for i in 1:length(scanning_solutions.solutions))
            end
            plot!(plt,min1,min2)
            plot!(plt,max1,max2)
        end
        label_str = "$(arg2) vs $(arg1)"
    end

    println(label_str)
    first_solution = true


    for sol in scanning_solutions.solutions
        # Only label the first line:
        local_label = first_solution ? label_str : ""
        first_solution = false

        # The DiffEqRecipes: pass `idxs=idxs` to choose which variables to plot
        #plot!(plt, sol; idxs=idxs, color=color, alpha=alpha, label=local_label)
    end
    return plt
end

"""
    solutions = solveMonteCarlo(sys, tspan, num_samples, var_dict=Dict())

Performs a Monte-Carlo parameter sampling over `var_dict`, drawing parameter
values from each interval uniformly at random. `num_samples` controls the number
of random draws performed. Returns an array of `ODESolution` objects.

# Arguments
- `sys`:       System definition (e.g., ModelingToolkit or similar).
- `tspan`:     A tuple `(tstart, tend)` specifying the simulation time interval.
- `num_samples`: Number of Monte-Carlo draws (random samples of parameters).
- `var_dict`:  Dictionary whose entries define each parameter's interval, e.g.
               `Dict(:p => 1 ± 0.1)`. If empty, defaults to `Dict()`.

# Optional keyword arguments
- `dt`:                    Timestep to use for `saveat` in the ODE solver. Defaults to 0.01.
- `interesting_variables`: Extra names/symbols of variables for post-processing or labeling.
                           Defaults to empty.

# Returns
- A `ScanningSolutions` (or similar) struct holding:
  - An array of ODESolution objects
  - Labels for the variables
  - Metadata about the scan
"""
function solveMonteCarlo(sys, tspan, num_samples; var_dict=Dict(), dt=0.01, interesting_variables=[])
    # 1) Get the intervals for each parameter from var_dict
    param_ranges = getIntervals(sys, var_dict) 
    # The number of random draws
    N = num_samples

    # Prepare a time grid for solution saving
    t_start, t_end = tspan
    timesteps = t_start:dt:t_end

    # Allocate space for the solutions
    solutions = Vector{ODESolution}(undef, N)

    for i in 1:N
        # Draw one random sample for each parameter
        param_dict = Dict()
        for (sym, inter) in param_ranges
            lo, hi = inf(inter), sup(inter)
            # Uniform draw from [lo, hi]
            param_dict[sym] = rand() * (hi - lo) + lo
        end

        # Build and solve the ODE problem
        prob = ODEProblem(sys, nothing, (t_start, t_end), param_dict)
        sol = solve(prob, saveat=timesteps)
        solutions[i] = sol

        println("Monte-Carlo sample ", i, " of ", N, ": done.")
    end

    # You can reuse or define a similar container for solutions as in the old code
    labels = Tuple([unknowns(sys)..., interesting_variables...])
    return ScanningSolutions(
        solutions,
        labels,
        Dict(
            :problem     => sys,
            :ts          => tspan,
            :num_samples => num_samples
        ),
        :monte
    )
end

