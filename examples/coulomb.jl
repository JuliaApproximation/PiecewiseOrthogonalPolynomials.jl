using PiecewiseOrthogonalPolynomials, ClassicalOrthogonalPolynomials, Plots
using Base: oneto

let M = 10
    global r = reverse(2.0 .^ (-(0:M))); r = [-reverse(r); 0; r]
    global V = x -> max(-1/abs(x), -2.0 ^(M))
end

Q = DirichletPolynomial(r)
P = ContinuousPolynomial{0}(Q)

a = expand(P, V)

A = P \ (a .* P)

R = P\Q
Δ = -weaklaplacian(Q)

p = 10; KR = Block.(oneto(p))

(R'*A*R)


plot(a)
