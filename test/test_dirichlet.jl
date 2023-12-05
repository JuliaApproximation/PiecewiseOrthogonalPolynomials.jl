using PiecewiseOrthogonalPolynomials, ClassicalOrthogonalPolynomials, ContinuumArrays, Test
using PiecewiseOrthogonalPolynomials: ArrowheadMatrix, plan_grid_transform

@testset "DirichletPolynomial" begin
    Q = DirichletPolynomial(range(-1,1; length=4))
    C = ContinuousPolynomial{1}(range(-1,1; length=4))
    P = ContinuousPolynomial{0}(range(-1,1; length=4))
    @test Q == Q
    @test Q ≠ P
    @test Q ≠ C
    @test P ≠ Q
    @test C ≠ Q

    @test PiecewisePolynomial(Q) == P
    @test PiecewisePolynomial(Q) ≠ Q
    @test Q ≠ PiecewisePolynomial(Q)


    f = expand(Q, x -> (1-x^2) * exp(x))
    @test f[0.1] ≈ (1-0.1^2) * exp(0.1)
    @test Q'Q isa Symmetric{Float64,<:ArrowheadMatrix}
    KR = Block.(Base.OneTo(10))
    @test ((Q'C) * (C\f))[KR] ≈ (Q'C)[KR,KR] * (C\f)[KR]

    @test (P * ((P\Q) * (Q\f)))[0.1] ≈ f[0.1]

    let x = 0.1
        @test diff(f)[x] ≈ -2exp(x)*x + exp(x)*(1 - x^2) 
    end

    Δ = weaklaplacian(Q)
    M = grammatrix(Q)
    @test (diff(Q)'diff(Q))[KR,KR] ≈ -Δ[KR,KR]
    @test (Q'Q)[KR,KR] ≈ M[KR,KR]

    @test (P / P \ f)[0.1] ≈ f[0.1]
    @test ((P \ C) * (C \ f))[KR] ≈ (P \ f)[KR]

    @testset "generic points" begin
        Q̃ = DirichletPolynomial(collect(Q.points))
        @test grammatrix(Q̃)[KR,KR] ≈ grammatrix(Q)[KR,KR]
    end

    @testset "plot" begin
        @test ClassicalOrthogonalPolynomials.grid(Q, 5) == ClassicalOrthogonalPolynomials.grid(Q, Block(2))
        @test ClassicalOrthogonalPolynomials.plotgrid(Q, 5) == ClassicalOrthogonalPolynomials.plotgrid(Q, Block(2))
    end

    @testset "expand" begin
        @test expand(sin.(f))[0.1] ≈ sin(f[0.1])
        @test expand(exp.(f))[0.1] ≈ exp(f[0.1])
    end

    @testset "transform" begin
        r = range(-1, 1; length=3)
        Q = DirichletPolynomial(r)

        x,F = plan_grid_transform(Q, Block(20));
        f = x -> (1-x^2) * exp(x)
        @test F*f.(x) ≈ transform(Q,f)[1:39]
        @test Q[0.1,1:39]' * (F*f.(x)) ≈ f(0.1)
        @test F\ (F*f.(x)) ≈ f.(x)

        f = (x,y) -> (1-x^2)*(1-y^2)*exp(x*cos(y))
        (x,y),F = ContinuumArrays.plan_grid_transform(Q, Block(20,21));
        vals = f.(x, reshape(y,1,1,size(y)...))
        @time V = F * vals;
        @test Q[0.1, Block.(1:20)]' * V * Q[0.2,Block.(1:21)] ≈ f(0.1,0.2)
        @test F \ V ≈ vals
    end
end