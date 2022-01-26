#!/usr/bin/env julia
"""
Simple reader routines for loading output data.

Prerequisites:
--------------
HDF5.jl
"""

using HDF5
using PyPlot
using DataFrames
pygui(:qt5)

using FastGaussQuadrature

include("Structures.jl")
include("ReadThornado.jl")

"""
Simple load routine for Athelas data.
"""
function Load_Output( Dir::AbstractString, filenumber::AbstractString; run::AbstractString = "RiemannProblem" )

  # fn :: String = string(Dir, "/", run, "_FluidFields_", filenumber, ".h5")
  # fn :: String = "../Example/Hydro/Executable/Athelas_MovingContact.h5"
  fn :: String = "../bin/athelas_Sod.h5"
  # fn :: String = "../Example/Hydro/Executable/Athelas_SmoothAdvection.h5"

  fid = h5open( fn, "r" )

  # Time
  time :: Float64 = 0.0

  x1   :: Array{Float64,1} = fid["/Spatial Grid/Grid"][:] 

  uCF  :: Matrix{Float64} = zeros( length(x1), 3 )
  uCF[:,1] = fid["/Conserved Fields/Specific Volume"][:]
  uCF[:,2] = fid["/Conserved Fields/Velocity"][:]
  uCF[:,3] = fid["/Conserved Fields/Specific Internal Energy"][:]

  uPF :: Matrix{Float64} = zeros( length(x1), 3 )
  uAF :: Matrix{Float64} = zeros( length(x1), 3 )

  state = State( time, x1, uCF, uPF, uAF )

  close( fid )

  return state

end


"""
Compute the n-point Gauss-Legendre quadrature.
"""
function ComputeQuadrature( nNodes::Int64 )
  nodes, weights = gausslegendre( nNodes );
  return nodes / 2.0, weights / 2.0
end


"""
Main cell average function.
"""
function CellAverage( Quantity::Array{Float64,1}, Weights::Array{Float64,1} )

  n  :: Int64 = length(Quantity)
  n2 :: Int64 = floor(Int, length(Quantity)/length(Weights))
  
  Avg    :: Array{Float64,1} = zeros( n2 )
  SumVar :: Float64 = 0.0
  iN     :: Int64 = 1
  iAvg   ::Int64 = 1
  @inbounds for iX in 1:n

    SumVar += Weights[iN] * Quantity[iX]


    iN += 1
    if ( iN > length(Weights) )
      Avg[iAvg] = SumVar
      iAvg += 1
      iN = 1
      SumVar = 0.0
    end

  end

  return Avg

end

"""
Cell average of quantity in nodal basis.
"""
function CellAverageData( Data::Matrix{Float64}, iCF::Int64, Weights::Array{Float64,1} )

  Quantity = Data[:,iCF]

  return CellAverage( Quantity, Weights )
end


"""
Compute pressure from conserved Lagrangian variables.
"""
function ComputePressure( Tau, Vel, EmT, GAMMA=1.4 )

  Em = EmT - 0.5 * Vel.*Vel
  Ev = Em ./ Tau
  P = (GAMMA - 1.0) .* Ev

  return P

end

# === testing ===

nNodes = 3
Data = Load_Output( "bleh", "000000" )

# reference, _ = load_thornado_single( ".", "000100", run="RiemannProblemSod" )

Nodes, Weights = ComputeQuadrature( nNodes )

Rad = Data.r
Tau = Data.uCF[:,1]
Vel = Data.uCF[:,2]
EmT = Data.uCF[:,3]

Em = EmT - 0.5 * Vel .* Vel

fig, ax = subplots()

ax.plot( Rad, 1.0 ./ Tau, marker=".", ls = " ", lw=2.0, label="Density", color="orchid" )
ax.plot( Rad, Vel, marker=".", ls=" ", lw=2.0, label="Velocity", color="tomato" )
# ax.plot( Rad, EmT, marker=" ", ls="--", lw=2.0, label="Total Specific Energy", color="teal" )
# ax.plot( Rad, ComputePressure(Tau, Vel, EmT), "o", ls="--", lw=1.0, label="Pressure" )

# ax.plot( reference.x1, reference.uCF_D, lw=1.0, label="Reference", color="black" )
# ax.plot( reference.x1, reference.uPF_V1, lw=1.0, label="Reference", color="black" )
ax.legend()

show()
