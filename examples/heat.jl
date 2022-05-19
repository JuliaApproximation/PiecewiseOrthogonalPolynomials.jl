using PiecewiseOrthogonalPolynomials, ClassicalOrthogonalPolynomials, DifferentialEquations, BlockArrays, Plots

r = range(-1,1; length=4)
C = ContinuousPolynomial{1}(r)
P = ContinuousPolynomial{0}(r)
x = axes(C,1)
D = Derivative(x)

K = Block.(1:20)
M = (C'C)[K,K]
Δ = BlockBandedMatrix((-(D*C)'*(D*C))[K,K])


c₀ = M \ ((C'P)*(P \ exp.(-30x.^2)))[K]

heat(u, (Δ, M), t) = M \ (Δ * u)

u = solve(ODEProblem(heat, c₀, (0.,1.), (Δ, M)), Tsit5(), reltol=1e-8, abstol=1e-8)

g = range(-1,1;length=100)
plot(g, C[g,K]*u(0.0))
plot!(g, C[g,K]*u(0.5))
plot!(g, C[g,K]*u(1))

s = P / P \ broadcast(x -> abs(x) ≤ 1/3 ? 1.0 : 0.5, x)

s .* P