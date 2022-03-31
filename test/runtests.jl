using PiecewiseSpectralMethods, ClassicalOrthogonalPolynomials, Test


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
    r = range(-1,1; length=2)
    r = range(0,1; length=4)
    P = ContinuousPolynomial{0}(r)
    C = ContinuousPolynomial{1}(r)
    P'P
    C'C
end

@testset "derivative" begin
    r = range(0,1; length=4)
    C = ContinuousPolynomial{1}(r)
    P = ContinuousPolynomial{0}(r)
    x = axes(C,1)
    D = Derivative(x)
    P\D*C
end