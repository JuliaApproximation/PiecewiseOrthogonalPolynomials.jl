module PiecewiseOrthogonalPolynomials
using ClassicalOrthogonalPolynomials, LinearAlgebra, BlockArrays, BlockBandedMatrices, BandedMatrices, ContinuumArrays, QuasiArrays, LazyArrays, LazyBandedMatrices, FillArrays, MatrixFactorizations, ArrayLayouts

import ArrayLayouts: sublayout, sub_materialize, symmetriclayout, transposelayout, SymmetricLayout, HermitianLayout, TriangularLayout, layout_getindex, materialize!, MatLdivVec, AbstractStridedLayout, triangulardata, MatMulMatAdd, MatMulVecAdd, _fill_lmul!
import BandedMatrices: _BandedMatrix
import BlockArrays: BlockSlice, block, blockindex, blockvec
import BlockBandedMatrices: _BandedBlockBandedMatrix, AbstractBandedBlockBandedMatrix, subblockbandwidths, blockbandwidths, AbstractBandedBlockBandedLayout, layout_replace_in_print_matrix
import ClassicalOrthogonalPolynomials: grid, plotgrid, ldiv, pad, adaptivetransform_ldiv, grammatrix, Plan, singularities, basis_singularities, singularitiesbroadcast
import ContinuumArrays: @simplify, factorize, TransformFactorization, AbstractBasisLayout, MemoryLayout, layout_broadcasted, ExpansionLayout, basis, plan_grid_transform, grammatrix
import LazyArrays: paddeddata
import LazyBandedMatrices: BlockBroadcastMatrix, BlockVec, BandedLazyLayouts, AbstractLazyBandedBlockBandedLayout, UpperOrLowerTriangular
import Base: axes, getindex, +, -, *, /, ==, \, OneTo, oneto, replace_in_print_matrix, copy, diff, getproperty, adjoint, transpose, tail, _sum
import LinearAlgebra: BlasInt
import MatrixFactorizations: reversecholcopy

export PiecewisePolynomial, ContinuousPolynomial, DirichletPolynomial, Derivative, Block, weaklaplacian, grammatrix

include("arrowhead.jl")
include("piecewisepolynomial.jl")
include("continuouspolynomial.jl")
include("dirichlet.jl")



end # module
