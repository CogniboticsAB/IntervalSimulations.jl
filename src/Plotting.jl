
"""
    plot_solution(result; idxs=(0,1), varnames=["x", "y"], kwargs...)

Dispatches to the correct plotting function based on result.kind[:type].
"""
function plot_solution(result::SimulationResult; idxs=(0,1), varnames=["x", "y"], kwargs...)
    plt = Plots.plot()
    plot_solution!(plt, result; idxs=idxs, varnames=varnames, kwargs...)
    return plt
end

"""
    plot_solution!(plt, result; idxs=(0,1), varnames=["x", "y"], kwargs...)

In-place plot based on simulation type.
"""
function plot_solution!(plt, result::SimulationResult; idxs=(0,1), varnames=["x", "y"], kwargs...)
    kind = result.kind[:type]
    vars = result.vars

    # Normalize idxs to integer indices
    if isa(idxs, MTK.Num)
        idxs = (0, findfirst(x -> isequal(x, idxs), vars))
    elseif isa(idxs, Int)
        idxs = (0, idxs)
    elseif isa(idxs, Tuple) && isa(idxs[2], MTK.Num)
        idxs = (idxs[1], findfirst(x -> isequal(x, idxs[2]), vars))
    elseif isa(idxs, Tuple{MTK.Num, MTK.Num})
        idxs = (
            findfirst(x -> isequal(x, idxs[1]), vars),
            findfirst(x -> isequal(x, idxs[2]), vars)
        )
    end

    if kind == :reach
        @info "Plotting Reachability solution"
        Plots.plot!(plt, result.sol; vars=idxs, label="", linealpha=0, fillalpha=0.3, kwargs...)

    elseif kind == :pce_bounded
        @info "Plotting PCE solution"
        i1, i2 = idxs
        if i1 == 0
            sym = vars[i2]
            m, l, u = result.sol[sym]
            Plots.plot!(plt, result.ts, l; label="Bounds PCE", linestyle=:dash, color=:red, kwargs...)
            Plots.plot!(plt, result.ts, u; label="", linestyle=:dash, color=:red, kwargs...)
        else
            s1, s2 = vars[i1], vars[i2]
            m1, l1, u1 = result.sol[s1]
            m2, l2, u2 = result.sol[s2]
            x = [u1...; reverse(l1)...; u1[1]]
            y = [u2...; reverse(l2)...; u2[1]]
            Plots.plot!(plt, x, y; seriestype=:shape, fillalpha=0.2, label=false, linecolor=:transparent)
            Plots.plot!(plt, m1, m2; label="PCE mean", color=:black, kwargs...)
        end
    elseif kind == :pce_bounded_split
        @info "Plotting PCE solution"
        i1, i2 = idxs
        if i1 == 0
            sym = vars[i2]
            m, l, u = result.sol[sym]
            Plots.plot!(plt, result.ts, l; label="Bounds PCE Split", color=:red, kwargs...)
            Plots.plot!(plt, result.ts, u; label="", color=:red, kwargs...)
        else
            s1, s2 = vars[i1], vars[i2]
            m1, l1, u1 = result.sol[s1]
            m2, l2, u2 = result.sol[s2]
            x = [u1...; reverse(l1)...; u1[1]]
            y = [u2...; reverse(l2)...; u2[1]]
            Plots.plot!(plt, x, y; seriestype=:shape, fillalpha=0.2, label=false, linecolor=:transparent)
            Plots.plot!(plt, m1, m2; label="PCE mean", color=:black, kwargs...)
        end

    elseif kind == :bounded_reach
        @info "Plotting bounded reach result"
        i1, i2 = idxs

        if i1 == 0
            # Plot over time for a single variable
            if haskey(result.sol, i2)
                lo, hi = result.sol[i2]
                Plots.plot!(plt, result.ts, lo; label="Bounds TM", color=:blue, linestyle=:dash, kwargs...)
                Plots.plot!(plt, result.ts, hi; label="", color=:blue, linestyle=:dash, kwargs...)
            else
                error("Variable index $i2 not found in result.sol")
            end
        else
            # Phase plot between two variables
            if haskey(result.sol, i1) && haskey(result.sol, i2)
                lo1, hi1 = result.sol[i1]
                lo2, hi2 = result.sol[i2]
                Plots.plot!(plt, lo1, lo2; label="Bounds TM", color=:blue, linestyle=:dash, kwargs...)
                Plots.plot!(plt, hi1, hi2; label="", color=:blue, linestyle=:dash, kwargs...)
            else
                error("Variable indices $i1 or $i2 not found in result.sol")
            end
        end
    elseif kind == :bounded_reach_split
        @info "Plotting bounded reach split result"
        i1, i2 = idxs

        if i1 == 0
            # Plot over time for a single variable
            if haskey(result.sol, i2)
                lo, hi = result.sol[i2]
                Plots.plot!(plt, result.ts, lo; label="Bounds TM Split", color=:blue, kwargs...)
                Plots.plot!(plt, result.ts, hi; label="", color=:blue, kwargs...)
            else
                error("Variable index $i2 not found in result.sol")
            end
        else
            # Phase plot between two variables
            if haskey(result.sol, i1) && haskey(result.sol, i2)
                lo1, hi1 = result.sol[i1]
                lo2, hi2 = result.sol[i2]
                Plots.plot!(plt, lo1, lo2; label="Bounds TM Split", color=:blue, kwargs...)
                Plots.plot!(plt, hi1, hi2; label="", color=:blue, kwargs...)
            else
                error("Variable indices $i1 or $i2 not found in result.sol")
            end
        end
    elseif kind in [:scanning_bounded, :monte_bounded]
        if kind == :scanning_bounded
            @info "Plotting scanning bounds"
            label = "Bounds SM"
            color = :green
        else
            @info "Plotting monte bounds"
            label = "Bounds MC"
            color = :purple
        end
        
        i1, i2 = isa(idxs, Tuple) ? idxs : (0, idxs)
    
        if i1 == 0
            # Time vs variable
            Plots.plot!(plt, result.ts, result.sol[Symbol("min$i2")]; label=label,color = color, kwargs...)
            Plots.plot!(plt, result.ts, result.sol[Symbol("max$i2")]; label="",color=color, kwargs...)
        else
            # Phase plot
            Plots.plot!(plt,
                result.sol[Symbol("min$i1")], result.sol[Symbol("min$i2")];
                label=label,color = color, kwargs...)
            Plots.plot!(plt,
                result.sol[Symbol("max$i1")], result.sol[Symbol("max$i2")];
                label="",color = color, kwargs...)
        end
        elseif kind in [:scanning_bounded, :monte_bounded]
        if kind == :scanning_bounded
            @info "Plotting scanning bounds"
            label = "Bounds SM"
        else
            @info "Plotting monte bounds"
            label = "Bounds MC"
        end
        
        i1, i2 = isa(idxs, Tuple) ? idxs : (0, idxs)
    
        if i1 == 0
            # Time vs variable
            Plots.plot!(plt, result.ts, result.sol[Symbol("min$i2")]; label=label, kwargs...)
            Plots.plot!(plt, result.ts, result.sol[Symbol("max$i2")]; label="", kwargs...)
        else
            # Phase plot
            Plots.plot!(plt,
                result.sol[Symbol("min$i1")], result.sol[Symbol("min$i2")];
                label=label, kwargs...)
            Plots.plot!(plt,
                result.sol[Symbol("max$i1")], result.sol[Symbol("max$i2")];
                label="", kwargs...)
        end
    else
        @warn "Unrecognized result type: $kind"
    end

    return plt
end
