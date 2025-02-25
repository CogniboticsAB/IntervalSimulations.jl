module IntervalSimulations

using ReachabilityAnalysis
using Symbolics
using SymbolicUtils
using ModelingToolkit
using ModelingToolkit: t_nounits as t
using IntervalArithmetic
using FastGaussQuadrature
using DifferentialEquations
using Base.Iterators: product

include("functions.jl")
export myOwnFunction

include("mtk2ivp.jl")
export runmtk

include("PCElegendre.jl")
export runlegendre

end
