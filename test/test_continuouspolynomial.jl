using PiecewiseOrthogonalPolynomials, StaticArrays, InfiniteArrays, ContinuumArrays, Test
using LazyBandedMatrices: MemoryLayout, AbstractBandedBlockBandedLayout, BlockVec
using ForwardDiff: derivative
using ContinuumArrays: plan_grid_transform

@testset "ContinuousPolynomial" begin
    @testset "transform" begin
        r = range(-1, 1; length=10)
        P = ContinuousPolynomial{0}(r)
        @test grid(P[:,Block.(Base.OneTo(3))]) == grid(PiecewisePolynomial(P)[:,Block.(Base.OneTo(3))]) == grid(ContinuousPolynomial{1}(r)[:,Block.(Base.OneTo(2))])
        @test grid(P[:,1:5]) == grid(PiecewisePolynomial(P)[:,1:5])
        @test grid(P[:,1:12]) == grid(PiecewisePolynomial(P)[:,1:12]) == grid(ContinuousPolynomial{1}(r)[:,1:5])
        x = axes(P,1)
        @test P[:,Block.(Base.OneTo(3))] \ x ≈ (P\ x)[Block.(1:3)]
        @test (P/P\x)[0.1] ≈ 0.1
        @test (P/P\exp.(x))[0.1] ≈ exp(0.1)
        g,F = ContinuumArrays.plan_grid_transform(P, Block(10));
        @test F * exp.(g) ≈ transform(P , exp)[Block.(1:10)]
        @test F \ (F * exp.(g)) ≈ exp.(g)

        C = ContinuousPolynomial{1}(r)
        g,F = ContinuumArrays.plan_grid_transform(C, Block(10));

        @test F * exp.(g) ≈ transform(C,exp)[1:91]
        @test C[0.1,Block.(1:10)]' * (F * exp.(g)) ≈ exp(0.1)
        @test F \ (F * exp.(g)) ≈ exp.(g)

        f = (x,y) -> exp(x*cos(y))
        (x,y),F = ContinuumArrays.plan_grid_transform(C, Block(10,11));
        vals = f.(x, reshape(y,1,1,size(y)...))
        @time V = F * vals;
        @test C[0.1, Block.(1:10)]' * V * C[0.2,Block.(1:11)] ≈ exp(0.1*cos(0.2))
        @test F \ V ≈ vals
    end

    @testset "lowering" begin
        c = [randn(20); zeros(∞)]
        for r in (range(-1, 1; length=2), range(0, 1; length=2), range(0, 1; length=4))
            P = ContinuousPolynomial{0}(r)
            C = ContinuousPolynomial{1}(r)
            R = P \ C
            for x in range(first(r), last(r); length=10)
                @test (C*c)[x] ≈ (P*@inferred(R*c))[x]
            end
        end
    end


    @testset "mass" begin
        for r in (range(-1,1; length=2), range(0,1; length=4), range(0, 1; length=4))
            P = ContinuousPolynomial{0}(r)
            C = ContinuousPolynomial{1}(r)

            @test P ≠ C
            @test C ≠ P
            @test P == PiecewisePolynomial(P)
            @test PiecewisePolynomial(P) == P
            @test C ≠ PiecewisePolynomial(P)
            @test PiecewisePolynomial(P) ≠ C

            JR = Block.(1:10)
            KR = Block.(1:11)
            R = P\C
            @test P[0.1,1:10]'*R[1:10,1:10] ≈ C[0.1,1:10]'
            @test R[KR,JR]'*((P'P)[KR,KR]*R[KR,JR]) ≈ (C'C)[JR,JR]
            @test (P'C)[JR,JR] ≈ (C'P)[JR,JR]'
        end

        @testset "collect versus range" begin
            for r in (range(-1,1; length=4), range(0,1; length=4))
                P = ContinuousPolynomial{0}(r)
                P̃ = ContinuousPolynomial{0}(collect(r))
                KR = Block.(1:5)
                @test grammatrix(P)[KR,KR] ≈ grammatrix(P̃)[KR,KR]

                C = ContinuousPolynomial{1}(r)
                C̃ = ContinuousPolynomial{1}(collect(r))
                @test grammatrix(C)[KR,KR] ≈ grammatrix(C̃)[KR,KR]
            end
        end
    end

    @testset "derivative" begin
        for r in (range(0,1; length=4), [0, 0.2, 0.5, 1])
            C = ContinuousPolynomial{1}(r)
            P = ContinuousPolynomial{0}(r)
            x = axes(C,1)
            D = Derivative(x)
            A = P\D*C

            @test (D*expand(C, exp))[0.1] ≈ exp(0.1)

            xx = rand(5)
            c = (x, p) -> ContinuousPolynomial{1, eltype(x)}(r)[x, p]

            for p = 1:30
                @test derivative.(x->c(x, p), xx) ≈ (P*A)[xx, p]
            end
        end

    end

    @testset "multiplication" begin
        r = range(-1,1; length=4)
        P = ContinuousPolynomial{0}(r)
        C = ContinuousPolynomial{1}(r)
        x = axes(P,1)
        a = P / P \ broadcast(x -> abs(x) ≤ 1/3 ? 1.0 : 0.5, x)
        @test (P \ (a .* P))[Block.(1:2), Block.(1:2)] ≈ Diagonal([0.5,1,0.5,0.5,1,0.5])

        c = [randn(4); Zeros(∞)]
        @test ((a .* C) * c)[0.1] ≈ (C*c)[0.1]
        @test ((a .* C) * c)[0.4] ≈ (C*c)[0.4]/2
    end

    @testset "weak Laplacian" begin
        for r in (range(0,1; length=5), range(-1,1; length=5))
            C = ContinuousPolynomial{1}(r)
            Δ = -diff(C)'*diff(C)
            KR = Block.(1:10)
            @test Δ[KR,KR] ≈ weaklaplacian(C)[KR,KR]

            M = C'C
            L = Δ + M

            @test MemoryLayout(Δ) isa AbstractBandedBlockBandedLayout
            @test MemoryLayout(L) isa AbstractBandedBlockBandedLayout

            KR = Block.(1:10)
            @test L[KR,KR] ≈ (weaklaplacian(C) + grammatrix(C))[KR,KR]
        end
    end

    @testset "conversion" begin
        r = range(0,1; length=5)
        C = ContinuousPolynomial{1}(r)
        P = ContinuousPolynomial{0}(r)
        c = [randn(5); zeros(∞)]
        g = C * c
        @test C \ g ≈ c
        g̃ = P / P \ g
        @test g[0.1] ≈ g̃[0.1]
        x = axes(C,1)
        e = C / C \ exp.(x)
        @test e[[0.1,0.7]] ≈ exp.([0.1,0.7])
    end

    @testset "static broadcast" begin
        C = ContinuousPolynomial{1}(range(0,1; length=4))
        x = SVector(0.1, 0.2)
        @test view(C, x, 1) .* view(C, x, 1) ≈ C[x,1] .* C[x,1]

        @test C \ (C[:,1] .* C[:,1]) ≈ [1; zeros(3); -1/2; zeros(∞)]
        @test C \ (C[:,1] .* C[:,3]) ≈ zeros(∞)
    end

    @testset "variable coefficients" begin
        r = range(-1,1; length=4)
        C = ContinuousPolynomial{1}(r)
        P = ContinuousPolynomial{0}(r)
        x = axes(P,1)
        D = Derivative(x)

        a = expand(P, x -> abs(x) ≤ 1/3 ? 2 : 3)
        L = (D*C)'* (a .* (D*C))
    end

    @testset "expand" begin
        r = range(-1,1; length=4)
        C = ContinuousPolynomial{1}(r)
        f = expand(C, exp)
        @test expand(exp.(f))[0.1] ≈ exp(exp(0.1))
    end

    @testset "sum" begin
        r = [-1, 0.1, 0.2, 1]
        P = ContinuousPolynomial{0}(r)
        C = ContinuousPolynomial{1}(r)
        @test sum(expand(P, exp)) ≈ sum(expand(C, exp)) ≈ ℯ - 1/ℯ
    end

    @testset "f_xy example" begin
        r = range(-1,1; length=3)
        P = ContinuousPolynomial{0}(r)
        M,N = Block(80),Block(81)
        ((x,y),pl) = plan_grid_transform(P, (M,N))
        f = (x,y) -> exp(x*cos(10y))
        F = f.(x, reshape(y, 1, 1, size(y)...))
        X = pl * F
        @test P[0.1,Block(1):M]' * X * P[0.2,Block(1):N] ≈ f(0.1,0.2)
        @test pl \ X ≈ F

        C = ContinuousPolynomial{1}(r)
        pl_C = plan_transform(C, (M-1,N-1))
        F_C = f.(x, reshape(y, 1, 1, size(y,1),:))
        X_C = pl_C * F_C

        @test C[0.1,Block(1):M-1]' * X_C * C[0.2,Block(1):N-1] ≈ f(0.1,0.2)

        D_x = (P \ diff(C))[Block(1):M, Block(1):M-1]
        D_y = (P \ diff(C))[Block(1):N, Block(1):N-1]

        f_xy = (x,y) -> -10sin(10y)*exp(x*cos(10y)) - 10cos(10y)*x*sin(10y)*exp(x*cos(10y))
        @test P[0.1,Block(1):M]'*(D_x*X_C*D_y')*P[0.2,Block(1):N] ≈ f_xy(0.1,0.2)

        # We can transform back to get the values on a large grid:

        F_xy = pl \ (D_x*X_C*D_y')

        @test F_xy ≈ f_xy.(x, reshape(y,1,1,size(y)...))
    end
end