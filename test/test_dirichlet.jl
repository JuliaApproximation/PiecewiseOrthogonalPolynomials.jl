using PiecewiseOrthogonalPolynomials, ClassicalOrthogonalPolynomials, Test
using PiecewiseOrthogonalPolynomials: ArrowheadMatrix

@testset "DirichletPolynomial" begin
    Q = DirichletPolynomial(range(-1,1; length=4))
    C = ContinuousPolynomial{1}(range(-1,1; length=4))
    P = ContinuousPolynomial{0}(range(-1,1; length=4))
    @test Q == Q
    @test Q ≠ P
    @test Q ≠ C
    @test P ≠ Q
    @test C ≠ Q


    f = expand(Q, x -> (1-x^2) * exp(x))
    @test f[0.1] ≈ (1-0.1^2) * exp(0.1)
    @test Q'Q isa Symmetric{Float64,<:ArrowheadMatrix}
    KR = Block.(Base.OneTo(10))
    @test ((Q'C) * (C\f))[KR] ≈ (Q'C)[KR,KR] * (C\f)[KR]

    let x = 0.1
        @test diff(f)[x] ≈ -2exp(x)*x + exp(x)*(1 - x^2) 
    end

    Δ = weaklaplacian(Q)
    M = grammatrix(Q)
    @test (diff(Q)'diff(Q))[KR,KR] ≈ -Δ[KR,KR]
    @test (Q'Q)[KR,KR] ≈ M[KR,KR]
end