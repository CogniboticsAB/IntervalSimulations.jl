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
using Plots
using Base: product
using Base.Threads
using ProgressBars
using IntervalOptimisation
using Unidecode

include("functions.jl")
export solveInterval, intervalPlot, intervalPlot!, solveScanning, plotScanning, plotScanning!

include("mtk2ivp.jl")
export createIVP, createIVP2


end
