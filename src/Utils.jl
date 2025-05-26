"""
    getIntervals(sys, pval)

Extracts the interval-valued parameters from `sys` and merges them with the
provided dictionary `pval`. All keys are converted to `Num` for consistency.
"""
function getIntervals(sys, pval::Dict)
    d = ModelingToolkit.defaults(sys)
    new_u = Dict(kv for kv in d if kv.second isa IntervalArithmetic.Interval)
    converted_new_u = Dict(Symbolics.Num(k) => v for (k, v) in new_u)
    #println(converted_new_u)
    intervals = merge(converted_new_u, pval)
    intervals2 = Dict(Symbolics.Num(k) => v for (k, v) in intervals)
    final_intervals = Dict(Symbolics.Num(k) => v for (k, v) in intervals if v isa IntervalArithmetic.Interval)
    not_intervals = Dict(k => v for (k, v) in intervals if !(v isa IntervalArithmetic.Interval))
    return final_intervals, not_intervals, intervals2
end

"""
    validate_interesting_variables(sys, interesting_vars)

Throws an error if any of the `interesting_vars` are actually parameters,
to prevent analysis confusion.
"""
function validate_interesting_variables(sys, interesting_vars)
    all_params = ModelingToolkit.parameters(sys)
    common_params = filter(x -> any(y -> isequal(x, y), all_params), interesting_vars)
    if !isempty(common_params)
        error("Interesting variables cannot be parameters. Problematic variables: ", common_params)
    end
end

"""
    merge_bounds(bounds1, bounds2, ...) -> SimulationResult

Merges multiple `SimulationResult`s containing bounds, taking the pointwise min/max
of all inputs. Assumes all bounds are over the same variables and time steps.
"""
function merge_bounds(first_bound::SimulationResult, rest_bounds::SimulationResult...)
    merged = first_bound

    for b in rest_bounds
        merged = merge_bounds_pairwise(merged, b)
    end

    return merged
end

# Helper function to merge two SimulationResults
function merge_bounds_pairwise(a::SimulationResult, b::SimulationResult)
    #@assert string.(a.vars) == string.(b.vars) "Variable sets must match"
    #@assert a.ts == b.ts "Time steps must match"

    vars = a.vars
    times = a.ts
    merged_data = Dict{Symbol, Vector{Float64}}()

    for i in 1:length(vars)
        min_a = a.sol[Symbol("min$i")]
        max_a = a.sol[Symbol("max$i")]
        min_b = b.sol[Symbol("min$i")]
        max_b = b.sol[Symbol("max$i")]

        merged_data[Symbol("min$i")] = min.(min_a, min_b)
        merged_data[Symbol("max$i")] = max.(max_a, max_b)
    end

    return SimulationResult(
        merged_data,
        vars,
        times,
        merge(a.kind, b.kind, Dict(:type => :merged_bounds))
    )
end
