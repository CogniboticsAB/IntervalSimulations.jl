"""
    getIntervals(sys, pval)

Extracts the interval-valued parameters from `sys` and merges them with the
provided dictionary `pval`. All keys are converted to `Num` for consistency.
"""
function getIntervals(sys, pval::Dict)
    d = ModelingToolkit.defaults(sys)
    new_u = Dict(kv for kv in d if kv.second isa IntervalArithmetic.Interval)
    converted_new_u = Dict(Symbolics.Num(k) => v for (k, v) in new_u)
    intervals = merge(converted_new_u, pval)
    final_intervals = Dict(Symbolics.Num(k) => v for (k, v) in intervals)
    return final_intervals
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
