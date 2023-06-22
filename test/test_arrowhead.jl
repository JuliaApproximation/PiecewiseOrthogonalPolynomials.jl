using PiecewiseOrthogonalPolynomials
using PiecewiseOrthogonalPolynomials: LeftArrowheadMatrix

C = ContinuousPolynomial{1}(range(-1,1; length=4))
P = ContinuousPolynomial{0}(range(-1,1; length=4))
M = P'P

x = 
D = Derivative(x)

(P\C)'

using InfiniteArrays, BlockArrays
using BandedMatrices: _BandedMatrix
import Base:oneto
T = Float64

v = (convert(T, 2):2:∞) ./ (3:2:∞)

N = length(P.points)

L = LeftArrowheadMatrix(
    [_BandedMatrix(Ones{T}(2, N-1)/2, oneto(N), 1, 0),
     _BandedMatrix(Vcat(-Ones{T}(1, N-1)/2, Ones{T}(1, N-1)/2), oneto(N), 1, 0)],
    Fill(_BandedMatrix(Hcat(v, Zeros{T}(∞), -v)', axes(v,1), 0, 2), N-1))
