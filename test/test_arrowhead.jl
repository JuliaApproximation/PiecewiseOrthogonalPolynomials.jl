using PiecewiseOrthogonalPolynomials, FillArrays, BandedMatrices, MatrixFactorizations, BlockBandedMatrices, Base64, Test
using PiecewiseOrthogonalPolynomials: ArrowheadMatrix
using InfiniteArrays, BlockArrays
using BandedMatrices: _BandedMatrix
import Base: oneto


@testset "ArrowheadMatrix" begin
    @testset "operators" begin
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

        @test blockbandwidths(L) == (1,1)
        @test subblockbandwidths(L) == (0,1)

        @test stringmime("text/plain", L[Block.(Base.OneTo(3)), Block.(Base.OneTo(3))]) == "3×3-blocked 9×10 ArrowheadMatrix{Float64, BandedMatrix{Float64, Fill{Float64, 2, Tuple{Base.OneTo{$Int}, Base.OneTo{$Int}}}, Base.OneTo{$Int}}, Vector{BandedMatrix{Float64, Fill{Float64, 2, Tuple{Base.OneTo{$Int}, Base.OneTo{$Int}}}, Base.OneTo{$Int}}}, Vector{BandedMatrix{Float64, LazyArrays.ApplyArray{Float64, 2, typeof(vcat), Tuple{Fill{Float64, 2, Tuple{Base.OneTo{$Int}, Base.OneTo{$Int}}}, Fill{Float64, 2, Tuple{Base.OneTo{$Int}, Base.OneTo{$Int}}}}}, Base.OneTo{$Int}}}, Vector{BandedMatrix{Float64, Matrix{Float64}, Base.OneTo{$Int}}}}:\n  0.5   0.5    ⋅    ⋅   │   0.666667    ⋅          ⋅        │   ⋅    ⋅    ⋅ \n   ⋅    0.5   0.5   ⋅   │    ⋅         0.666667    ⋅        │   ⋅    ⋅    ⋅ \n   ⋅     ⋅    0.5  0.5  │    ⋅          ⋅         0.666667  │   ⋅    ⋅    ⋅ \n ───────────────────────┼───────────────────────────────────┼───────────────\n -0.5   0.5    ⋅    ⋅   │   0.0         ⋅          ⋅        │  0.8   ⋅    ⋅ \n   ⋅   -0.5   0.5   ⋅   │    ⋅         0.0         ⋅        │   ⋅   0.8   ⋅ \n   ⋅     ⋅   -0.5  0.5  │    ⋅          ⋅         0.0       │   ⋅    ⋅   0.8\n ───────────────────────┼───────────────────────────────────┼───────────────\n   ⋅     ⋅     ⋅    ⋅   │  -0.666667    ⋅          ⋅        │  0.0   ⋅    ⋅ \n   ⋅     ⋅     ⋅    ⋅   │    ⋅        -0.666667    ⋅        │   ⋅   0.0   ⋅ \n   ⋅     ⋅     ⋅    ⋅   │    ⋅          ⋅        -0.666667  │   ⋅    ⋅   0.0"
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