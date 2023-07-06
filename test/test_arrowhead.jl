using PiecewiseOrthogonalPolynomials, FillArrays, BandedMatrices
using PiecewiseOrthogonalPolynomials: LeftArrowheadMatrix
using InfiniteArrays, BlockArrays
using BandedMatrices: _BandedMatrix
import Base: oneto

C = ContinuousPolynomial{1}(range(-1,1; length=4))
P = ContinuousPolynomial{0}(range(-1,1; length=4))

T = Float64

v = (convert(T, 2):2:∞) ./ (3:2:∞)

N = length(P.points)

L = LeftArrowheadMatrix(
    [_BandedMatrix(Ones{T}(2, N)/2, oneto(N-1), 0, 1),
     _BandedMatrix(Vcat(Ones{T}(1, N)/2, -Ones{T}(1, N)/2), oneto(N-1), 0, 1)],
    Fill(_BandedMatrix(Hcat(v, Zeros{T}(∞), -v)', axes(v,1), 2, 0), N-1))

KR = Block.(Base.OneTo(5))
@test (P\C)[KR,KR] == L[KR,KR]

C'C


diff(C; dims=1)'diff(C; dims=1)

M = C'C



n = 3; p = 5;
A = LeftArrowheadMatrix([BandedMatrix(0 => randn(n) .+ 10, 1 => randn(n-1), -1 => randn(n-1)), 
                      BandedMatrix((0 => randn(n-1), 1 => randn(n-1)), (n-1,n)), 
                      BandedMatrix((0 => randn(n-1), 1 => randn(n-1)), (n-1,n))],
                     fill(BandedMatrix((-1 => randn(p) .+ 10, -3 => randn(p-2)), (p+1, p)), n-1))

