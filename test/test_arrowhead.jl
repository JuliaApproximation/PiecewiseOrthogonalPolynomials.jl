using PiecewiseOrthogonalPolynomials, FillArrays
using PiecewiseOrthogonalPolynomials: LeftArrowheadMatrix
using InfiniteArrays, BlockArrays
using BandedMatrices: _BandedMatrix
import Base: oneto

C = ContinuousPolynomial{1}(range(-1,1; length=4))
P = ContinuousPolynomial{0}(range(-1,1; length=4))
M = P'P

(P\C)


T = Float64

v = (convert(T, 2):2:∞) ./ (3:2:∞)

N = length(P.points)

L = LeftArrowheadMatrix(
    [_BandedMatrix(Ones{T}(2, N)/2, oneto(N-1), 0, 1),
     _BandedMatrix(Vcat(Ones{T}(1, N)/2, -Ones{T}(1, N)/2), oneto(N-1), 0, 1)],
    Fill(_BandedMatrix(Hcat(v, Zeros{T}(∞), -v)', axes(v,1), 2, 0), N-1))

KR = Block.(Base.OneTo(5))
@test (P\C)[KR,KR] == L[KR,KR]