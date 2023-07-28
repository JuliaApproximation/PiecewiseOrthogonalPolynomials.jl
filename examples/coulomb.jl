using PiecewiseOrthogonalPolynomials, ClassicalOrthogonalPolynomials, BandedMatrices, Plots
using Base: oneto

let M = 10
    global r = reverse(2.0 .^ (-(0:M))); r = [-reverse(r); 0; r]
    global V = x -> max(-1/abs(x), -2.0 ^(M))
end

Q = DirichletPolynomial(r)
P = ContinuousPolynomial{0}(Q)

a = expand(P, V)



p = 10; KR = Block.(oneto(p))
Δ = (diff(Q)'diff(Q))[KR,KR];
M = (Q'Q)[KR,KR]



A = P \ (a .* P)
R = P\Q
@time M = (R'* (P'P) *  A*R)[KR,KR];

λ,V = eigen((Matrix(Δ)), (Matrix(M)));

@assert isreal(V[:,end])


plot!(Q[:,KR] * real(V[:,end-2]))

g = range(-1,1; length=1000)
Pl = Q[g,KR];

p = plot(a; ylims=(-30,1), legend=false)
scatter!(r, zero(r))
for j = size(V,2):-1:size(V,2)-10; plot!(g, Pl * real(V[:,j]) .+ λ[j]) end
p


r = range(-1,1; length=10)
Q = DirichletPolynomial(r)
λ,V = eigen(Matrix((-weaklaplacian(Q))[KR,KR]), Matrix(grammatrix(Q)[KR,KR]))

plot(Q[:,KR] * V[:,4])





ContinuousPolynomial{0}(r) \ ContinuousPolynomial{1}(r)


