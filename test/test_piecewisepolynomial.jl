using PiecewiseOrthogonalPolynomials, ClassicalOrthogonalPolynomials, FillArrays, Test
using PiecewiseOrthogonalPolynomials: plan_grid_transform, grid
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
            for P in (PiecewisePolynomial(Chebyshev(), range(0,1; length=3)), PiecewisePolynomial(Legendre(), range(0,1; length=3)))
                @test P[:,Block.(Base.OneTo(3))] \ P[:,1:2] == Eye(6,2)
                @test P[:,Block.(Base.OneTo(3))] \ (P[:,2] .* P[:,1:2]) ≈ P[:,Block.(Base.OneTo(3))] \ (P[:,2] .* P[:,Block(1)]) ≈ [P[:,Block.(Base.OneTo(3))]\(P[:,2] .* P[:,1]) P[:,Block.(Base.OneTo(3))]\(P[:,2] .* P[:,2])]
                
                
                x,F = plan_grid_transform(P, (Block(10),2), 1)
                KR = Block.(1:10)
                @test F * [exp.(x) ;;; cos.(x)] ≈ [transform(P,exp)[KR] transform(P,cos)[KR]]
                @test F \ (F * [exp.(x) ;;; cos.(x)]) ≈ [exp.(x) ;;; cos.(x)]

                t = axes(P,1)
                @test (P \ [cos.(t) sin.(t)])[KR,:] ≈ [(P\cos.(t))[KR,:] (P\sin.(t))[KR,:]]

                F = plan_transform(P, (Block(10), Block(11)))
                x,y = grid(P, Block(10)), grid(P, Block(11))
                C = F * exp.(x .+ cos.(reshape(y,1,1,size(y)...)))
                @test P[0.1, Block.(1:10)]' * C * P[0.2, Block.(1:11)] ≈ exp(0.1 + cos(0.2))
            end
        end
    end

    @testset "expand" begin
        r = range(-1, 1; length=4)
        P = PiecewisePolynomial(Legendre(), r)
        f = expand(P, exp)
        @test expand(exp.(f))[0.1] ≈ exp(f[0.1])
    end

    @testset "inv transform" begin
        r = range(-1, 1; length=4)
        for P in (PiecewisePolynomial(Chebyshev(), r), PiecewisePolynomial(Legendre(), r))
            pl = plan_transform(P, Block(10))
            @test size(pl) == (10,3)
            X = randn(size(pl))
            @test pl\(pl*X) ≈ X

            pl = plan_transform(P, Block(10,11))
            @test size(pl) == (10,3, 11, 3)
            X = randn(size(pl))
            @test pl\(pl*X) ≈ X
        end
    end

    @testset "tensor transform" begin
        for P in (PiecewisePolynomial(Chebyshev(), range(0,1; length=3)), PiecewisePolynomial(Legendre(), range(0,1; length=3)))
            xy, pl = plan_grid_transform(P, (Block(10),Block(11), 3), (1,2))
            @test size(pl) == (10, 2, 11, 2, 3)
            x,y=first(xy),last(xy)
            y = reshape(y,1,1,size(y)...)

            X = [exp.(x .+ cos.(y)) ;;;;; exp.(y .+ sin.(x)) ;;;;; cos.(x .+ sin.(y))]
            C = pl*X

            @test P[0.1, Block.(1:10)]' * C[:,:,1] * P[0.2, Block.(1:11)] ≈ exp(0.1 + cos(0.2))
            @test P[0.1, Block.(1:10)]' * C[:,:,2] * P[0.2, Block.(1:11)] ≈ exp(0.2 + sin(0.1))
            @test P[0.1, Block.(1:10)]' * C[:,:,3] * P[0.2, Block.(1:11)] ≈ cos(0.1 + sin(0.2))

            @test pl\C ≈ X
        end
    end
end