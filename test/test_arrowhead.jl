using PiecewiseOrthogonalPolynomials, FillArrays, BandedMatrices, MatrixFactorizations, BlockBandedMatrices, Base64, Test
using PiecewiseOrthogonalPolynomials: ArrowheadMatrix
using InfiniteArrays, BlockArrays
using BandedMatrices: _BandedMatrix
import Base: oneto


@testset "ArrowheadMatrix" begin
    @testset "operators" begin
        C = ContinuousPolynomial{1}(range(-1,1; length=4))
        P = ContinuousPolynomial{0}(range(-1,1; length=4))
        P̃ = ContinuousPolynomial{0}(collect(range(-1,1; length=4)))

        L = P\C
        @test L isa ArrowheadMatrix
        KR = Block.(Base.OneTo(5))
        @test (P̃\C)[KR,KR] == @inferred(L[KR,KR])
        @test blockbandwidths(L) == (1,1)
        @test subblockbandwidths(L) == (0,1)

        @test stringmime("text/plain", L[Block.(Base.OneTo(3)), Block.(Base.OneTo(3))]) == "3×3-blocked 9×10 ArrowheadMatrix{Float64, BandedMatrix{Float64, Fill{Float64, 2, Tuple{Base.OneTo{$Int}, Base.OneTo{$Int}}}, Base.OneTo{$Int}}, Tuple{BandedMatrix{Float64, Fill{Float64, 2, Tuple{Base.OneTo{$Int}, Base.OneTo{$Int}}}, Base.OneTo{$Int}}}, Tuple{BandedMatrix{Float64, LazyArrays.ApplyArray{Float64, 2, typeof(vcat), Tuple{Fill{Float64, 2, Tuple{Base.OneTo{$Int}, Base.OneTo{$Int}}}, Fill{Float64, 2, Tuple{Base.OneTo{$Int}, Base.OneTo{$Int}}}}}, Base.OneTo{$Int}}}, Vector{BandedMatrix{Float64, Matrix{Float64}, Base.OneTo{$Int}}}}:\n  0.5   0.5    ⋅    ⋅   │   0.666667    ⋅          ⋅        │   ⋅    ⋅    ⋅ \n   ⋅    0.5   0.5   ⋅   │    ⋅         0.666667    ⋅        │   ⋅    ⋅    ⋅ \n   ⋅     ⋅    0.5  0.5  │    ⋅          ⋅         0.666667  │   ⋅    ⋅    ⋅ \n ───────────────────────┼───────────────────────────────────┼───────────────\n -0.5   0.5    ⋅    ⋅   │   0.0         ⋅          ⋅        │  0.8   ⋅    ⋅ \n   ⋅   -0.5   0.5   ⋅   │    ⋅         0.0         ⋅        │   ⋅   0.8   ⋅ \n   ⋅     ⋅   -0.5  0.5  │    ⋅          ⋅         0.0       │   ⋅    ⋅   0.8\n ───────────────────────┼───────────────────────────────────┼───────────────\n   ⋅     ⋅     ⋅    ⋅   │  -0.666667    ⋅          ⋅        │  0.0   ⋅    ⋅ \n   ⋅     ⋅     ⋅    ⋅   │    ⋅        -0.666667    ⋅        │   ⋅   0.0   ⋅ \n   ⋅     ⋅     ⋅    ⋅   │    ⋅          ⋅        -0.666667  │   ⋅    ⋅   0.0"
    end

    @testset "reversecholesky" begin
        n = 3; p = 5;
        A = ArrowheadMatrix(BandedMatrix(0 => randn(n) .+ 10, 1 => randn(n-1), -1 => randn(n-1)), 
                                [BandedMatrix((0 => randn(n-1), -1 => randn(n-1)), (n,n-1)) for _=1:2],
                                BandedMatrix{Float64, Matrix{Float64}, Base.OneTo{Int}}[],
                            fill(BandedMatrix((0 => randn(p) .+ 10, 2 => randn(p-2)), (p, p)), n-1))

        @test reversecholesky(Symmetric(Matrix(A))).U ≈ reversecholesky!(Symmetric(copy(A))).U


        A = ArrowheadMatrix(BandedMatrix(0 => randn(n) .+ 10, 1 => randn(n-1), -1 => randn(n-1)), 
                                [BandedMatrix((0 => randn(n-1), -1 => randn(n-1)), (n,n-1)) for _=1:2],
                                BandedMatrix{Float64, Matrix{Float64}, Base.OneTo{Int}}[],
                            fill(BandedMatrix((0 => randn(p) .+ 10, 1 => randn(p-1), 2 => randn(p-2)), (p, p)), n-1))


        @test reversecholesky(Symmetric(Matrix(A))).U ≈ reversecholesky!(Symmetric(copy(A))).U
    end
end

using ClassicalOrthogonalPolynomials
using ClassicalOrthogonalPolynomials: grammatrix
using BlockBandedMatrices: _BandedBlockBandedMatrix
using LazyBandedMatrices

function weaklaplacian(r)
    N = length(r)
    s = step(r)
    t1 = Vcat((N-1)/2, Fill((N-1), N-2), (N-1)/2)
    t2 = Fill(-(N-1)/2, N-1)
    ArrowheadMatrix(LazyBandedMatrices.SymTridiagonal(t1, t2), BandedMatrix{Float64, Matrix{Float64}, Base.OneTo{Int}}[], BandedMatrix{Float64, Matrix{Float64}, Base.OneTo{Int}}[],
        Fill(Diagonal(16 .* (1:∞) .^ 2 ./ (s .* ((2:2:∞) .+ 1))), N-1))
end

r = range(-1,1; length=4)
mats = weaklaplacian(r)
C = ContinuousPolynomial{1}(r)
KR = Block.(1:5)
@test (diff(C)'diff(C))[KR,KR] ≈ mats[KR,KR]


function mass_matrix_arrow_head(r)
    T = eltype(r)

    N = length(r) - 1

    a11 = SymTridiagonal(vcat(2/(3N), fill(4/(3N), N-1), 2/(3N)), fill(1/(3N), N))
    a21 = _BandedMatrix(Fill(2/(3N), 2, N+1), N, 0, 1)
    a31 = _BandedMatrix(Vcat(Fill(4/(15N), 1, N+1), Fill(-4/(15N), 1, N+1)), N, 0, 1)
    
    A = BlockVcat(a11, a21, a31)
    BlockHcat(A, BlockVcat(a21', Zeros{T}(N,N), Zeros{T}(N,N)), BlockVcat(a31', Zeros{T}(N,N), Zeros{T}(N,N)))
end

function mass_matrix_bubble(j, r)
    N = length(r) - 1
    
    d = grammatrix(Legendre()).diag
    L = (Legendre() \ Weighted(Jacobi(1,1))).args[2].dv

    a = j > 2 ? -d[j]*L[j-2]*L[j] : 0.0
    [a / N; L[j]^2 * (d[j] + d[j+2]) / N; -d[j+2]*L[j]*L[j+2]/ N]
end

r = range(-1,1; length=3);
C = ContinuousPolynomial{1}(r);
M = C' * C

T = Float64
convert(T,2) ./ (1:2:∞)

L = (2:2:∞) ./ (3:2:∞)

T = Float64
a = ((4:4:∞) .* (-2:2:∞)) ./ ((1:2:∞) .* (3:2:∞) .* (-1:2:∞))
b = (((2:2:∞) ./ (3:2:∞)).^2 .* (convert(T,2) ./ (1:2:∞) .+ convert(T,2) ./ (5:2:∞)))

c = (convert(T,2) ./ (5:2:∞)) .* ((2:2:∞) ./ (3:2:∞)) .* ((6:2:∞) ./ (7:2:∞))

T = eltype(r)

N = length(r) - 1

a11 = SymTridiagonal(vcat(2/(3N), fill(4/(3N), N-1), 2/(3N)), fill(1/(3N), N))
a21 = _BandedMatrix(Fill(2/(3N), 2, N), N+1, 1, 0)
a31 = _BandedMatrix(Vcat(Fill(-4/(15N), 1, N), Fill(4/(15N), 1, N)), N+1, 1, 0)


M̃ = Symmetric(ArrowheadMatrix(a11, [a21, a31],
    BandedMatrix{Float64, Matrix{Float64}, Base.OneTo{Int}}[],
    Fill(_BandedMatrix(Vcat((-a/N)',
     Zeros(1,∞),
    (b/N)'), ∞, 0, 2), N)))


S = Symmetric(parent(M̃)[Block.(oneto(5)), Block.(oneto(5))])
@time reversecholesky!(S)

@mass_matrix_bubble(3, r)[1] ≈ M[Block(2), Block(4)][1,1] ≈ M[Block(2), Block(4)][2,2] ≈ M[Block(2), Block(4)][3,3]


mass_matrix_bubble(3, r)[2] ≈ M[Block(4), Block(4)][1,1] ≈ M[Block(4), Block(4)][2,2] ≈ M[Block(4), Block(4)][3,3]
mass_matrix_bubble(3, r)[3] ≈ M[Block(6), Block(4)][1,1] ≈ M[Block(6), Block(4)][2,2] ≈ M[Block(6), Block(4)][3,3]


@test mass_matrix_arrow_head(r)[Block.(1:3), Block(1)] ≈ (C' * C)[Block.(1:3), Block(1)]

r = range(-1,1; length=5);
C = ContinuousPolynomial{1}(r);
mass_matrix_arrow_head(r)[Block.(1:3), Block(1)] ≈ (C' * C)[Block.(1:3), Block(1)]