#!/usr/bin/env julia
"""
Structures for holding output data

Prerequisites:
--------------
None
"""

struct State

  time :: Float64

  r   :: Array{Float64}

  uCF :: Matrix{Float64}
  uPF :: Matrix{Float64}
  uAF :: Matrix{Float64}
  
  SLopeLimiter :: Array{Int64}

end