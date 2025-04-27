module IntervalSimulations

# Core dependencies
import ReachabilityAnalysis
import ReachabilityAnalysis: @ivp
import Symbolics
import SymbolicUtils
import ModelingToolkit
import IntervalArithmetic
import FastGaussQuadrature
import DifferentialEquations
import TaylorModels
import Plots
import ProgressBars
import Base.Threads: @threads
import Base.Iterators: product
import Unidecode
import LazySets
import SymbolicIndexingInterface: parameter_values, state_values
import SciMLStructures: Tunable, canonicalize, replace, replace!
import Distributions

import IntervalArithmetic: ±, inf, sup, Interval
export ±, inf, sup, Interval

# Use short aliases internally (optional)
const RA  = ReachabilityAnalysis
const MTK = ModelingToolkit
const IA  = IntervalArithmetic
const TM  = TaylorModels
const plt = Plots
const DE = DifferentialEquations

# Include internal submodules (no `module` keyword inside files)
include("Types.jl")
include("Utils.jl")
include("PCE.jl")
include("ReachabilityTools.jl")
include("ScanningMonteCarlo.jl")
include("Plotting.jl")
include("MTKPreparation.jl")  # your former mtk2ivp.jl

# Public API
export
    # types
    SimulationResult,

    # solvers
    solve_pce,
    solve_reachability,
    solve_parameter_scan,
    solve_monte_carlo,
    solve_pce_split,
    solve_reachability_split,

    # postprocessing
    calculate_bounds_pce,
    calculate_bounds_split,
    get_bounds,
    compute_bounds,
    calculate_bounds_split_reach,

    # plotting
    plot_solution,
    plot_solution!,

    # utils
    getIntervals,
    validate_interesting_variables,

    # modeling toolkit support
    createIVP,

    poly_coeff_l2norms,
    evaluate_bounds_over_time,
    print_sorted_coeffs,
    merge_bounds,
    coeff_energy_by_order

end

