using PiecewiseOrthogonalPolynomials, FillArrays, BandedMatrices
using PiecewiseOrthogonalPolynomials: ArrowheadMatrix
using InfiniteArrays, BlockArrays
using BandedMatrices: _BandedMatrix
import Base: oneto

C = ContinuousPolynomial{1}(range(-1,1; length=4))
P = ContinuousPolynomial{0}(range(-1,1; length=4))

T = Float64

v = (convert(T, 2):2:∞) ./ (3:2:∞)

N = length(P.points)

L = ArrowheadMatrix(_BandedMatrix(Ones{T}(2, N)/2, oneto(N-1), 0, 1),
     [_BandedMatrix(Fill(v[1], 1, N-1), oneto(N-1), 0, 0)],
     [_BandedMatrix(Vcat(Ones{T}(1, N)/2, -Ones{T}(1, N)/2), oneto(N-1), 0, 1)],
    Fill(_BandedMatrix(Hcat(v, Zeros{T}(∞), -v)', axes(v,1), 1, 1), N-1))

KR = Block.(Base.OneTo(5))
@test (P\C)[KR,KR] == L[KR,KR]

C'C


# diff(C; dims=1)'diff(C; dims=1)
# M = C'C



n = 3; p = 5;
A = ArrowheadMatrix(BandedMatrix(0 => randn(n) .+ 10, 1 => randn(n-1), -1 => randn(n-1)), 
                        [BandedMatrix((0 => randn(n-1), -1 => randn(n-1)), (n,n-1)) for _=1:2],
                        BandedMatrix{Float64}[],
                     fill(BandedMatrix((0 => randn(p) .+ 10, 2 => randn(p-2)), (p, p)), n-1))

cholesky(Symmetric(Matrix(A)))
