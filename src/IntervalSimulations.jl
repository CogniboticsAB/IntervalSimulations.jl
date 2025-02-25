module IntervalSimulations

using ReachabilityAnalysis
using Symbolics
using SymbolicUtils
using ModelingToolkit
using ModelingToolkit: t_nounits as t

include("functions.jl")
export myOwnFunction

include("mtk2ivp.jl")
export runmtk

end
