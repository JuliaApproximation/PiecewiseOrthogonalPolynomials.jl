using PiecewiseOrthogonalPolynomials, FillArrays, BandedMatrices, MatrixFactorizations, BlockBandedMatrices, Base64, ClassicalOrthogonalPolynomials, Test
using PiecewiseOrthogonalPolynomials: BBBArrowheadMatrix
using InfiniteArrays, BlockArrays
using BandedMatrices: _BandedMatrix
import Base: oneto, OneTo


@testset "BBBArrowheadMatrix" begin
    @testset "Constructor" begin
        n = 4; p = 5;
        @test BBBArrowheadMatrix{Float64}(BandedMatrix(0 => 1:n, 1 => 1:n-1, -1 => 1:n-1),
                                ntuple(_ -> BandedMatrix((0 => randn(n-1), -1 => randn(n-1)), (n,n-1)), 2),
                                ntuple(_ -> BandedMatrix((0 => randn(n), 1 => randn(n-1)), (n-1,n)), 3),
                            fill(BandedMatrix((0 => randn(p) .+ 10, 2 => randn(p-2), -1=> randn(p-1)), (p, p)), n-1)) isa BBBArrowheadMatrix{Float64}
    end

    @testset "Algebra" begin
        n = 4; p = 5;
        A = BBBArrowheadMatrix(BandedMatrix(0 => randn(n) .+ 10, 1 => randn(n-1), -1 => randn(n-1)),
                                ntuple(_ -> BandedMatrix((0 => randn(n-1), -1 => randn(n-1)), (n,n-1)), 2),
                                ntuple(_ -> BandedMatrix((0 => randn(n), 1 => randn(n-1)), (n-1,n)), 3),
                            fill(BandedMatrix((0 => randn(p) .+ 10, 2 => randn(p-2), -1=> randn(p-1)), (p, p)), n-1))

        @test 2A isa BBBArrowheadMatrix
        @test A*2 isa BBBArrowheadMatrix
        @test 2\A isa BBBArrowheadMatrix
        @test A/2 isa BBBArrowheadMatrix
        @test A+A isa BBBArrowheadMatrix
        @test A-A isa BBBArrowheadMatrix

        @test 2A == A*2 == A+A == 2Matrix(A)
        @test all(iszero,A-A)
        @test A + A' == A' + A == Matrix(A) + Matrix(A)'
        @test A - A' == Matrix(A) - Matrix(A)'
        @test A' - A == Matrix(A)' - Matrix(A)
        @test A/2 == 2\A == Matrix(A)/2
    end

    @testset "mul" begin
        n = 4; p = 5;
        A = BBBArrowheadMatrix(BandedMatrix(0 => randn(n) .+ 10, 1 => randn(n-1), -1 => randn(n-1)),
                                ntuple(_ -> BandedMatrix((0 => randn(n-1), -1 => randn(n-1)), (n,n-1)), 2),
                                ntuple(_ -> BandedMatrix((0 => randn(n), 1 => randn(n-1)), (n-1,n)), 3),
                            fill(BandedMatrix((0 => randn(p) .+ 10, 2 => randn(p-2), -1=> randn(p-1)), (p, p)), n-1))

        x = randn(size(A,1))
        X = randn(size(A))
        X̃ = randn(size(A,1),5)
        @test A*x ≈ Matrix(A)*x
        @test A'*x ≈ Matrix(A)'*x
        @test A*X ≈ Matrix(A)*X
        @test A*X̃ ≈ Matrix(A)*X̃
        @test X*A ≈ X*Matrix(A)
    end

    @testset "UpperTriangular" begin
        n = 4; p = 5;
        A = BBBArrowheadMatrix(BandedMatrix(0 => randn(n) .+ 10, 1 => randn(n-1), -1 => randn(n-1)),
                                [BandedMatrix((0 => randn(n-1), -1 => randn(n-1)), (n,n-1)) for _=1:2],
                                [BandedMatrix((0 => randn(n), 1 => randn(n-1)), (n-1,n)) for _=1:2],
                            fill(BandedMatrix((0 => randn(p) .+ 10, 2 => randn(p-2), -1=> randn(p-1)), (p, p)), n-1))

        c = randn(size(A,1))
        for T in (UpperTriangular(A), UnitUpperTriangular(A), LowerTriangular(A), UnitLowerTriangular(A),
                  UpperTriangular(A)')
            @test T \ c ≈ Matrix(T) \ c
            @test c' / T ≈ c' / Matrix(T)
        end
        for Typ in (UpperTriangular, UnitUpperTriangular)
            @test Typ(A).A == Typ(A.A)
            @test Typ(A).B == A.B
            @test isempty(Typ(A).C)
            @test Typ(A).D == map(Typ,A.D)
        end
        for Typ in (LowerTriangular, UnitLowerTriangular)
            @test Typ(A).A == Typ(A.A)
            @test isempty(Typ(A).B)
            @test Typ(A).C == A.C
            @test Typ(A).D == map(Typ,A.D)
        end
    end

    @testset "reversecholesky" begin
        n = 3; p = 5;
        A = BBBArrowheadMatrix(BandedMatrix(0 => randn(n) .+ 10, 1 => randn(n-1), -1 => randn(n-1)),
                                [BandedMatrix((0 => randn(n-1), -1 => randn(n-1)), (n,n-1)) for _=1:2],
                                BandedMatrix{Float64, Matrix{Float64}, OneTo{Int}}[],
                            fill(BandedMatrix((0 => randn(p) .+ 10, 2 => randn(p-2)), (p, p)), n-1))

        @test reversecholesky(Symmetric(Matrix(A))).U ≈ reversecholesky!(Symmetric(copy(A))).U


        A = BBBArrowheadMatrix(BandedMatrix(0 => randn(n) .+ 10, 1 => randn(n-1), -1 => randn(n-1)),
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
        @test L isa BBBArrowheadMatrix
        KR = Block.(OneTo(5))
        @test (P̃\C)[KR,KR] == @inferred(L[KR,KR])
        @test blockbandwidths(L) == (1,1)
        @test subblockbandwidths(L) == (1,1)

        @test stringmime("text/plain", L[Block.(OneTo(3)), Block.(OneTo(3))]) == "3×3-blocked 9×10 BBBArrowheadMatrix{Float64}:\n  0.5   0.5    ⋅    ⋅   │   0.333333    ⋅          ⋅        │   ⋅    ⋅    ⋅ \n   ⋅    0.5   0.5   ⋅   │    ⋅         0.333333    ⋅        │   ⋅    ⋅    ⋅ \n   ⋅     ⋅    0.5  0.5  │    ⋅          ⋅         0.333333  │   ⋅    ⋅    ⋅ \n ───────────────────────┼───────────────────────────────────┼───────────────\n -0.5   0.5    ⋅    ⋅   │   0.0         ⋅          ⋅        │  0.2   ⋅    ⋅ \n   ⋅   -0.5   0.5   ⋅   │    ⋅         0.0         ⋅        │   ⋅   0.2   ⋅ \n   ⋅     ⋅   -0.5  0.5  │    ⋅          ⋅         0.0       │   ⋅    ⋅   0.2\n ───────────────────────┼───────────────────────────────────┼───────────────\n   ⋅     ⋅     ⋅    ⋅   │  -0.333333    ⋅          ⋅        │  0.0   ⋅    ⋅ \n   ⋅     ⋅     ⋅    ⋅   │    ⋅        -0.333333    ⋅        │   ⋅   0.0   ⋅ \n   ⋅     ⋅     ⋅    ⋅   │    ⋅          ⋅        -0.333333  │   ⋅    ⋅   0.0"
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

        x = M[KR,KR] * transform(C, exp)[KR]
        @time c = F \ x;

        @test_broken c isa PseudoBlockArray # TODO: overload copy_similar in BlockArrays.jl

        @test (C[:,KR] * c)[0.1] ≈ 1.1952730862177243
    end

    @testset "Dirichlet" begin
        n = 5; p = 5;
        A = BBBArrowheadMatrix(BandedMatrix(0 => randn(n-2) .+ 10, 1 => randn(n-3), -1 => randn(n-3)),
                                [BandedMatrix((0 => randn(n-2), 1 => randn(n-2)), (n-2,n-1)) for _=1:2],
                                BandedMatrix{Float64, Matrix{Float64}, OneTo{Int}}[],
                            fill(BandedMatrix((0 => randn(p) .+ 10, 2 => randn(p-2)), (p, p)), n-1))
        U = reversecholesky(Symmetric(A)).U
        @test U ≈ reversecholesky(Matrix(Symmetric(A))).U
        @test U*U' ≈ Symmetric(A)
    end
end
