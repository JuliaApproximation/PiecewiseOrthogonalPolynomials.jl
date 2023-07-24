using PiecewiseOrthogonalPolynomials, ClassicalOrthogonalPolynomials, BlockArrays, Test, FillArrays, LinearAlgebra, StaticArrays, ContinuumArrays, Random
import Base: OneTo
import LazyBandedMatrices: MemoryLayout, AbstractBandedBlockBandedLayout, BlockVec
import ForwardDiff: derivative

Random.seed!(0)

include("test_arrowhead.jl")
include("test_piecewisepolynomial.jl")
include("test_continuouspolynomial.jl")
include("test_dirichlet.jl")
