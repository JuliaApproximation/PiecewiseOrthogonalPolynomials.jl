using PiecewiseOrthogonalPolynomials, ClassicalOrthogonalPolynomials, FillArrays, Test
using LazyBandedMatrices: BlockVec

@testset "PiecewisePolynomial" begin
    @testset "transform" begin
        for r in (range(-1, 1; length=2), range(-1, 1; length=4), range(0, 1; length=4)), T in (Chebyshev(), Legendre())
            P = PiecewisePolynomial(T, r)
            x = axes(P,1)
            @test P[:,Block.(Base.OneTo(3))] \ x isa BlockVec
            @test P[:,Block.(Base.OneTo(3))] \ x ≈ (P\ x)[Block.(1:3)]
            @test (P/P\x)[0.1] ≈ 0.1
            @test (P/P\exp.(x))[0.1] ≈ exp(0.1)
        end

        r = range(-1, 1; length=10_000)
        P = PiecewisePolynomial(Chebyshev(), r); Pₙ = P[:,Block.(1:1000)]; x = axes(P,1)
        @time u = Pₙ / Pₙ \ cos.(10_000x.^2);
        @test u[0.1] ≈ cos(10_000*0.1^2)

        r = range(-1, 1; length=10)
        P = PiecewisePolynomial(Chebyshev(), r); Pₙ = P[:,Block.(1:3)]; x = axes(P,1)
        @test grid(P[:,1:27]) == grid(Pₙ)

        @testset "matrix" begin
            P = PiecewisePolynomial(Chebyshev(), range(0,1; length=3))
            @test P[:,Block.(Base.OneTo(3))] \ P[:,1:2] == Eye(6,2)
            @test P[:,Block.(Base.OneTo(3))] \ (P[:,2] .* P[:,1:2]) ≈ P[:,Block.(Base.OneTo(3))] \ (P[:,2] .* P[:,Block(1)]) ≈ [P[:,Block.(Base.OneTo(3))]\(P[:,2] .* P[:,1]) P[:,Block.(Base.OneTo(3))]\(P[:,2] .* P[:,2])] 
        end
    end

    @testset "expand" begin
        r = range(-1, 1; length=4)
        P = PiecewisePolynomial(Legendre(), r)
        @test expand(exp.(f))[0.1] ≈ exp(f[0.1])
    end
end