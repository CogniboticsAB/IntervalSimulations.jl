module DifferentialEquationsExt

using IntervalSimulations
import DifferentialEquations

# Export the DifferentialEquations namespace if needed
if VERSION ≥ v"1.11.0" && isdefined(IntervalSimulations, :DE) &&
   nameof(IntervalSimulations.DE) == :DifferentialEquations
    # use DE-based features
end


# Optional: redefine `solve` behavior or add extra methods
# For example:
# IntervalSimulations.solve_de_based(...) = ...

end
