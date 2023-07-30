using PiecewiseOrthogonalPolynomials, ClassicalOrthogonalPolynomials, BandedMatrices, Plots
using Base: oneto

let M = 20, R = 100
    global r = reverse(2.0 .^ (-(0:M))); r = R*[-reverse(r); 0; r]
    global V = x -> max(-1/abs(x), -2.0 ^(M)/R)
end

Q = DirichletPolynomial(r)
P = ContinuousPolynomial{0}(Q)

a = expand(P, V)



p = 20; KR = Block.(oneto(p))
Δ = (diff(Q)'diff(Q))[KR,KR];
M = grammatrix(Q)[KR,KR];

R = P\Q
@time A = (R'* P' * (a .* P) *R)[KR,KR];


λ,V = eigen((Matrix(Δ) + 10A), (Matrix(M)))



g = range(-100,100; length=10000)
p = plot(g, 10a[g]; legend=false, ylims=(-150,5), linestyle=:dash, linewidth=2)
scatter!(r, 10a[r])
savefig("coulombpotential.pdf")

p = plot(g, 10a[g]; legend=false, ylims=(-30,5), linestyle=:dash, linewidth=2)
Pl = Q[g,KR];
for j = 1:100; plot!(g, Pl * real(V[:,j]) .+ λ[j], linewidth=2) end
p

savefig("coulomb.pdf")

g = range(-5,5; length=1000)
p = plot(g, 10a[g]; legend=false, ylims=(-40,0), linestyle=:dash, linewidth=2)
Pl = Q[g,KR];
for j = 1:10; plot!(g, Pl * real(V[:,j]) .+ λ[j], linewidth=2) end
p

savefig("coulomb.pdf")

p2 = plot(g, 10a[g]; legend=false, ylims=(-2800,-2700), linestyle=:dash, linewidth=2)

nanabs = x -> iszero(x) ? NaN : abs(x)
scatter(nanabs.(10coefficients(a)[1:size(V,1)]); yscale=:log10, legend=false)

KR = Block.(oneto(30)); scatter(10nanabs.(coefficients(expand(P[:,KR], V))[KR]); yscale=:log10, legend=false, yticks=10.0 .^(-20:5:5))
savefig("coulombcoeffs.pdf")
scatter!(nanabs.(V[:,2]); yscale=:log10)


a = expand(P, x -> x^2)
Δ = -weaklaplacian(Q)[KR,KR]
M = grammatrix(Q)[KR,KR]

R = P\Q
@time A = (R'* P' * (a .* P) *R)[KR,KR];
λ,V = eigen((Matrix(Δ) + 1000A), (Matrix(M)))


plot(Q[:,KR] * ((Δ + 1000A) \ [1; zeros(size(Δ,1)-1)]))
p


r = range(-1,1; length=10)
Q = DirichletPolynomial(r)
λ,V = eigen(Matrix((-weaklaplacian(Q))[KR,KR]), Matrix(grammatrix(Q)[KR,KR]))

plot(Q[:,KR] * V[:,4])





ContinuousPolynomial{0}(r) \ ContinuousPolynomial{1}(r)


