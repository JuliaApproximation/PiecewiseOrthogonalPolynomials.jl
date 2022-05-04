using PiecewiseOrthogonalPolynomials, ClassicalOrthogonalPolynomials, BlockArrays, Test


@testset "transform" begin
    for r in (range(-1, 1; length=2), range(-1, 1; length=4), range(0, 1; length=4)), T in (Chebyshev(), Legendre())
        P = PiecewisePolynomial(T, r)
        x = axes(P,1)
        @test P[:,Block.(Base.OneTo(3))] \ x ≈ (P\ x)[Block.(1:3)]
        @test (P/P\x)[0.1] ≈ 0.1
        @test (P/P\exp.(x))[0.1] ≈ exp(0.1)
    end

    r = range(-1, 1; length=10_000)
    P = PiecewisePolynomial(Chebyshev(), r); Pₙ = P[:,Block.(1:1000)]; x = axes(P,1)
    @time u = Pₙ / Pₙ \ cos.(10_000x.^2);
    @test u[0.1] ≈ cos(10_000*0.1^2)

    P = ContinuousPolynomial{0}(r)
    x = axes(P,1)
    @test P[:,Block.(Base.OneTo(3))] \ x ≈ (P\ x)[Block.(1:3)]
    @test (P/P\x)[0.1] ≈ 0.1
    @test (P/P\exp.(x))[0.1] ≈ exp(0.1)    
end

@testset "lowering" begin
    c = [randn(20); zeros(∞)]
    for r in (range(-1, 1; length=2), range(0, 1; length=2), range(0, 1; length=4))
        P = ContinuousPolynomial{0}(r)
        C = ContinuousPolynomial{1}(r)
        R = P \ C
        for x in range(first(r), last(r); length=100)
            @test (C*c)[x] ≈ (P*(R*c))[x]
        end
    end
end

@testset "mass" begin
    for r in (range(-1,1; length=2), range(0,1; length=4), range(0, 1; length=4))
        P = ContinuousPolynomial{0}(r)
        C = ContinuousPolynomial{1}(r)
        JR = Block.(1:10)
        KR = Block.(1:11)
        R = P\C
        @test R[KR,JR]'*((P'P)[KR,KR]*R[KR,JR]) ≈ (C'C)[JR,JR]
        @test (P'C)[JR,JR] ≈ (C'P)[JR,JR]'
        
    end
end

@testset "derivative" begin
    r = range(0,1; length=4)
    C = ContinuousPolynomial{1}(r)
    P = ContinuousPolynomial{0}(r)
    x = axes(C,1)
    D = Derivative(x)
    P\D*C
end

@testset "weak Laplacian" begin
    r = range(0,1; length=4)
    C = ContinuousPolynomial{1}(r)
    x = axes(C,1)
    D = Derivative(x)
    L = -(D*C)'*(D*C) + C'C
    KR = Block.(1:10)
    @time L[KR,KR]
end