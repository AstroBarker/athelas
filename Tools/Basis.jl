#!/usr/bin/env julia
"""
Tools for managing Taylor basis functions

Prerequisites:
--------------
FastGaussQuadrature.jl
"""

"""
Compute the n-point Gauss-Legendre quadrature.
Scaled to interval [-0.5, +0.5]
"""
function ComputeQuadrature(nNodes::Int64)
  nodes, weights = gausslegendre(nNodes)
  return nodes / 2.0, weights / 2.0
end


"""
Evaluate Basis
"""
function BasisEval(Basis::BasisType, Data::State, iCF::Int64, iX::Int64,
                   iEta::Int64)
  order::Int64 = Basis.order

  result::Float64 = 0.0
  for k in 1:order
    @inbounds result += Data.uCF[k, iX, iCF] * Basis.Phi[k, iX, iEta]
  end
  return result
end


"""
Integrate on an element
"""
function Integrate( Basis::BasisType, Data::State, Weights::Array{Float64, 1}, 
                    iCF::Int64, iX::Int64 )
  result :: Float64 = 0.0
  for iN in 1:length(Weights)
    result += Weights[iN] * BasisEval( Basis, Data, iCF, iX, iN + 1 )
  end
  return result
end