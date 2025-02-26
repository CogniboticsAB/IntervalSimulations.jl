include("PCElegendre.jl")

struct IntervalSolution
    sol  # either a ReachSolution or the dict-of-means/lower/upper from Legendre
    vars # a list of variable names/symbols
    ts   # time vector, if applicable
    kind # either :reach or :pce
end

struct ScanningSolutions
    solutions::Vector{ODESolution}
    varnames::Vector{String}
end


##############################################################################
# INTERVAL
##############################################################################

function solveInterval(prob, tspan, pval; dt=0.01, dim=4, solver=TMJets21a())
    if prob isa ODESystem
        println("Legendre")
        # Suppose runlegendre returns (solutionDict, timeVector), or something similar
        pce_sol, ts = runlegendre(prob, dim, tspan, dt, pval, Dict())
        # You can store the unknowns(prob) in a variables list if you want
        vars = unknowns(prob)
        return IntervalSolution(pce_sol, vars, ts, :pce)
    elseif (prob isa Tuple && prob[1] isa InitialValueProblem)
        println("IVP / Reachability")
        ra_sol = ReachabilityAnalysis.solve(prob[1], tspan=(tspan[1], tspan[2]), solver)
        save_times = tspan[1]:dt:tspan[2]
        # prob[2] might be the symbol list or variable set you want
        vars = prob[2]
        return IntervalSolution(ra_sol, vars, save_times, :reach)
    else
        error("solveInterval: unrecognized problem type")
    end
end

function intervalPlot(intervalsol::IntervalSolution; idxs=(0,1))
    fig = plot()
    intervalPlot!(fig, intervalsol; idxs=idxs)
    return fig
end

function intervalPlot!(fig, intervalsol::IntervalSolution; idxs=(0,1))
    if intervalsol.kind == :reach
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

    elseif intervalsol.kind == :pce
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

        _plotPCE!(fig, intervalsol, idxs)

    end
    return fig
end

# Helper to handle (0, i) or (i, j) or 1:3, etc.
function _plotPCE!(fig, intervalsol::IntervalSolution, idxs)
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
                plot!(fig, t_vec, mvec, label="mean $(var_symbol)")
                plot!(fig, t_vec, lvec, label="low $(var_symbol)", ls=:dash)
                plot!(fig, t_vec, uvec, label="high $(var_symbol)", ls=:dash)
            else
                # "phase plot" of var i2 vs var i1 (only the means)
                var_sym1 = vars[i1]
                var_sym2 = vars[i2]
                m1, l1, u1 = sol_dict[var_sym1]
                m2, l2, u2 = sol_dict[var_sym2]
                # For a phase plot, we might only plot the mean curve (unless you want intervals in 2D).
                plot!(fig, m1, m2, label="mean $(var_sym2) vs $(var_sym1)")
                plot!(fig, l1, l2, label="low $(var_sym2) vs $(var_sym1)")
                plot!(fig, u1, u2, label="high $(var_sym2) vs $(var_sym1)")
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
            plot!(fig, t_vec, mvec, label="mean $(var_symbol)")
            plot!(fig, t_vec, lvec, label="low $(var_symbol)", ls=:dash)
            plot!(fig, t_vec, uvec, label="high $(var_symbol)", ls=:dash)
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
function solveScanning(sys, tspan, grid_size, var_dict)
    # 1) Parameter intervals as Interval; e.g. var_dict[sym] => 1 ± 0.1
    #    We'll build a range(...) out of each.
    param_ranges = var_dict
    fine_grid_points = grid_size

    dt = 0.01
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
        sol = solve(prob, Rodas5(), saveat=timesteps)
        solutions[i] = sol

        println("Progress: ", round(i/total*100, digits=2), " %")
    end
    labels = [string(v) for v in unknowns(sys)]
    return ScanningSolutions(solutions, labels)
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
    if idxs[1] == 0
        arg1 = "t"
        arg2 = scanning_solutions.varnames[idxs[2]]
    else
        arg1 = scanning_solutions.varnames[idxs[1]]
        arg2 = scanning_solutions.varnames[idxs[2]]
    end

    label_str = "$(arg2) vs $(arg1)"
    first_solution = true

    for sol in scanning_solutions.solutions
        # Only label the first line:
        local_label = first_solution ? label_str : ""
        first_solution = false

        # The DiffEqRecipes: pass `idxs=idxs` to choose which variables to plot
        plot!(plt, sol; idxs=idxs, color=color, alpha=alpha, label=local_label)
    end
    return plt
end
