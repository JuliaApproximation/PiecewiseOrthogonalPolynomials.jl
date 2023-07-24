using PiecewiseOrthogonalPolynomials, ClassicalOrthogonalPolynomials, Test

@testset "DirichletPolynomial" begin
    Q = DirichletPolynomial(range(-1,1; length=4))
    C = ContinuousPolynomial{1}(range(-1,1; length=4))
    P = ContinuousPolynomial{0}(range(-1,1; length=4))
    @test Q == Q

    f = expand(Q, x -> (1-x^2) * exp(x))
    @test f[0.1] ≈ (1-0.1^2) * exp(0.1)
    @test Q'Q isa Symmetric{Float64,<:ArrowheadMatrix}
    KR = Block.(Base.OneTo(20))
    ((Q'C) * (C\f))[KR] ≈ (Q'C)[KR,KR] * (C\f)[KR]
end