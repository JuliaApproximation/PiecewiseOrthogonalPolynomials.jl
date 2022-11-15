using PiecewiseOrthogonalPolynomials, ClassicalOrthogonalPolynomials, DifferentialEquations, BlockArrays, BlockBandedMatrices, Plots
import ClassicalOrthogonalPolynomials: transform

heat(u, (Δ, M), t) = M \ (Δ * u)

r = range(-1,1; length=4)
C = ContinuousPolynomial{1}(r)
P = ContinuousPolynomial{0}(r)
x = axes(C,1)
D = Derivative(x)

K = Block.(1:20)
M = cholesky(Symmetric((C'C)[K,K]))
Δ = (-(D*C)'*(D*C))[K,K]


c₀ = M \ (C'expand(P, x -> exp(-30x^2)))[K]

u = solve(ODEProblem(heat, c₀, (0.,1.), (Δ, M)), Tsit5(), reltol=1e-8, abstol=1e-8)


# Here we implement Trap Rule with fixed times step
# M*u_t = Δ*u 
# becomes (M - h/2*Δ) \ (M + h/2*Δ)

K = Block.(1:100)

Δ = (-(D*C)'*(D*C))[K,K]
M = (C'C)[K,K]
h = 0.001
A = (M - h/2*Δ);
B = M + h/2*Δ

c₀ = Vector(transform(C, x -> if x < -1/3
        exp(x+1/3) 
    elseif abs(x) ≤ 1/3
    9x^2
    else
        exp(x-1/3)
    end)[K])

u = [c₀]
for k = 1:100
    push!(u, A \ (B *  last(u)))
end

g = range(-1,1;length=1000)
p = plot(g, C[g,K]*u[1]; legend=false)
for k = 1:10
    plot!(g, C[g,K]*u[10k+1])
end; p 


p = plot(g, C[g,K]*u[end]; legend=false)



A = factorize(M - h*Δ);

u = [c₀]
for k = 1:100
    push!(u, A \ (M *  last(u)))
end
plot!(g, C[g,K]*u[end]; legend=false)

g = range(-1,1;length=1000)
p = plot(g, C[g,K]*u[1]; legend=false)
for k = 1:10
    plot!(g, C[g,K]*u[k+1])
end; p 

# u(t,x) = C[x,:] * c(t)

# different speeds
# ⟨v,u_t⟩ = -⟨∇v, a(x)∇*u⟩

K = Block.(1:20)
a = expand(P, x -> abs(x) ≤ 1/3 ? 0.5 : 1.0)
L = (-(D*C)'*(a .* (D*C)))[K,K]

u = solve(ODEProblem(heat, c₀, (0.,1.), (L, M)), Tsit5(), reltol=1e-8, abstol=1e-8)

u = solve(ODEProblem(heat, c₀, (0.,1.), (L, M)), Tsit5(), reltol=1e-8, abstol=1e-8)

g = range(-1,1;length=100)
p = plot()
for t in range(0,1; length=4)
    plot!(g, C[g,K]*u(t))
end; p


wave(up, u, (Δ, M), t) = M \ (Δ * u)

u = solve(SecondOrderODEProblem(wave, zero(c₀), c₀, (0.,1.), (L, M)), Tsit5(), reltol=1e-8, abstol=1e-8)

g = range(-1,1;length=100)
p = plot()
for t in range(0,1; length=10)
    plot!(g, C[g,K]*u(t).x[2])
end; p


