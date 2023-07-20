using PiecewiseOrthogonalPolynomials, FillArrays, BandedMatrices, MatrixFactorizations, BlockBandedMatrices, Base64, ClassicalOrthogonalPolynomials, Test
using PiecewiseOrthogonalPolynomials: ArrowheadMatrix
using InfiniteArrays, BlockArrays
using BandedMatrices: _BandedMatrix
import Base: oneto, OneTo


@testset "ArrowheadMatrix" begin
    @testset "UpperTriangular" begin
        n = 3; p = 5;
        A = ArrowheadMatrix(BandedMatrix(0 => randn(n) .+ 10, 1 => randn(n-1), -1 => randn(n-1)), 
                                [BandedMatrix((0 => randn(n-1), -1 => randn(n-1)), (n,n-1)) for _=1:2],
                                BandedMatrix{Float64, Matrix{Float64}, OneTo{Int}}[],
                            fill(BandedMatrix((0 => randn(p) .+ 10, 2 => randn(p-2)), (p, p)), n-1))
    end

    @testset "reversecholesky" begin
        n = 3; p = 5;
        A = ArrowheadMatrix(BandedMatrix(0 => randn(n) .+ 10, 1 => randn(n-1), -1 => randn(n-1)), 
                                [BandedMatrix((0 => randn(n-1), -1 => randn(n-1)), (n,n-1)) for _=1:2],
                                BandedMatrix{Float64, Matrix{Float64}, OneTo{Int}}[],
                            fill(BandedMatrix((0 => randn(p) .+ 10, 2 => randn(p-2)), (p, p)), n-1))

        @test reversecholesky(Symmetric(Matrix(A))).U ≈ reversecholesky!(Symmetric(copy(A))).U


        A = ArrowheadMatrix(BandedMatrix(0 => randn(n) .+ 10, 1 => randn(n-1), -1 => randn(n-1)), 
                                [BandedMatrix((0 => randn(n-1), -1 => randn(n-1)), (n,n-1)) for _=1:2],
                                BandedMatrix{Float64, Matrix{Float64}, OneTo{Int}}[],
                            fill(BandedMatrix((0 => randn(p) .+ 10, 1 => randn(p-1), 2 => randn(p-2)), (p, p)), n-1))


        @test reversecholesky(Symmetric(Matrix(A))).U ≈ reversecholesky!(Symmetric(copy(A))).U
    end

    @testset "operators" begin
        C = ContinuousPolynomial{1}(range(-1,1; length=4))
        P = ContinuousPolynomial{0}(range(-1,1; length=4))
        P̃ = ContinuousPolynomial{0}(collect(range(-1,1; length=4)))

        L = P\C
        @test L isa ArrowheadMatrix
        KR = Block.(OneTo(5))
        @test (P̃\C)[KR,KR] == @inferred(L[KR,KR])
        @test blockbandwidths(L) == (1,1)
        @test subblockbandwidths(L) == (1,1)

        @test stringmime("text/plain", L[Block.(OneTo(3)), Block.(OneTo(3))]) == "3×3-blocked 9×10 ArrowheadMatrix{Float64, BandedMatrix{Float64, Fill{Float64, 2, Tuple{OneTo{$Int}, OneTo{$Int}}}, OneTo{$Int}}, Tuple{BandedMatrix{Float64, Fill{Float64, 2, Tuple{OneTo{$Int}, OneTo{$Int}}}, OneTo{$Int}}}, Tuple{BandedMatrix{Float64, LazyArrays.ApplyArray{Float64, 2, typeof(vcat), Tuple{Fill{Float64, 2, Tuple{OneTo{$Int}, OneTo{$Int}}}, Fill{Float64, 2, Tuple{OneTo{$Int}, OneTo{$Int}}}}}, OneTo{$Int}}}, Vector{BandedMatrix{Float64, Matrix{Float64}, OneTo{$Int}}}}:\n  0.5   0.5    ⋅    ⋅   │   0.666667    ⋅          ⋅        │   ⋅    ⋅    ⋅ \n   ⋅    0.5   0.5   ⋅   │    ⋅         0.666667    ⋅        │   ⋅    ⋅    ⋅ \n   ⋅     ⋅    0.5  0.5  │    ⋅          ⋅         0.666667  │   ⋅    ⋅    ⋅ \n ───────────────────────┼───────────────────────────────────┼───────────────\n -0.5   0.5    ⋅    ⋅   │   0.0         ⋅          ⋅        │  0.8   ⋅    ⋅ \n   ⋅   -0.5   0.5   ⋅   │    ⋅         0.0         ⋅        │   ⋅   0.8   ⋅ \n   ⋅     ⋅   -0.5  0.5  │    ⋅          ⋅         0.0       │   ⋅    ⋅   0.8\n ───────────────────────┼───────────────────────────────────┼───────────────\n   ⋅     ⋅     ⋅    ⋅   │  -0.666667    ⋅          ⋅        │  0.0   ⋅    ⋅ \n   ⋅     ⋅     ⋅    ⋅   │    ⋅        -0.666667    ⋅        │   ⋅   0.0   ⋅ \n   ⋅     ⋅     ⋅    ⋅   │    ⋅          ⋅        -0.666667  │   ⋅    ⋅   0.0"
    end

    @testset "Helmholtz solve" begin
        r = range(-1,1; length=4)
        C = ContinuousPolynomial{1}(r)
        Δ = weaklaplacian(C)
        M = grammatrix(C)

        KR = Block.(1:5)
        @test -(diff(C)'diff(C))[KR,KR] ≈ Δ[KR,KR]

        KR = Block.(oneto(100))
        @time F = reversecholesky(Symmetric(parent(-Δ+M)[KR,KR]));
        c = F \ (M[KR,KR] * transform(C, exp)[KR]);

        @test (C[:,KR] * c)[0.1] ≈ 1.1952730862177243
    end
end
