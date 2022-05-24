using PiecewiseOrthogonalPolynomials, ClassicalOrthogonalPolynomials, DifferentialEquations, BlockArrays, BlockBandedMatrices, Plots

heat(u, (Δ, M), t) = M \ (Δ * u)

r = range(-1,1; length=4)
C = ContinuousPolynomial{1}(r)
P = ContinuousPolynomial{0}(r)
x = axes(C,1)
D = Derivative(x)

K = Block.(1:40)
M = (C'C)[K,K]
Δ = BlockBandedMatrix((-(D*C)'*(D*C))[K,K])


c₀ = M \ ((C'P)*(P \ exp.(-30x.^2)))[K]

u = solve(ODEProblem(heat, c₀, (0.,1.), (Δ, M)), Tsit5(), reltol=1e-8, abstol=1e-8)

g = range(-1,1;length=100)
plot(g, C[g,K]*u(0.0))
plot!(g, C[g,K]*u(0.5))
plot!(g, C[g,K]*u(1))

# u(t,x) = C[x,:] * c(t)

# a(x) * u_tt = Δ*u
# C' * (a. * C) * c_tt = -((D*C') * D*C) * c

K = Block.(1:20)
a = P / P \ broadcast(x -> abs(x) ≤ 1/3 ? 0.5 : 1.0, x)
M̃ = (C'*(a .* C))[K,K]
M = (C'C)[K,K]
Δ = BlockBandedMatrix((-(D*C)'*(D*C))[K,K])

c₀ = M \ ((C'P)*(P \ exp.(-30x.^2)))[K]

u = solve(ODEProblem(heat, c₀, (0.,1.), (Δ, M̃)), Tsit5(), reltol=1e-8, abstol=1e-8)

g = range(-1,1;length=100)
p = plot()
for t in range(0,1; length=4)
    plot!(g, C[g,K]*u(t))
end; p


wave(up, u, (Δ, M), t) = M \ (Δ * u)

u = solve(SecondOrderODEProblem(wave, zero(c₀), c₀, (0.,1.), (Δ, M̃)), Tsit5(), reltol=1e-8, abstol=1e-8)

g = range(-1,1;length=100)
p = plot()
for t in range(0,1; length=10)
    plot!(g, C[g,K]*u(t).x[2])
end; p
