C = ContinuousPolynomial{1}(range(-1,1; length=2))
F = ContinuousPolynomial{1}(range(-1,1; length=3))

n = 50

R = zeros(length(axes(F,2)[Block.(1:n)]), length(axes(C,2)[Block.(1:n)]))
for k = 1:size(R,2) R[:,k] = (F \ C[:,k])[Block.(1:n)] end

x = axes(C,1)
D = Derivative(x)

Δ_c = (-(D*C)'D*C)[Block.(1:n), Block.(1:n)]
Δ_f = (-(D*F)'D*F)[Block.(1:n), Block.(1:n)]
M_c = (C'C)[Block.(1:n), Block.(1:n)]
M_f = (F'F)[Block.(1:n), Block.(1:n)]

Δ_c = (-(D*C)'D*C)[Block.(1:n), Block.(1:n)]
Δ_f = (-(D*F)'D*F)[Block.(1:n), Block.(1:n)]


eigvals(Matrix(Δ_c), Matrix(M_c))

eigvals(Matrix(Δ_f), Matrix(M_f))

using Plots
v_c = C[:,Block.(1:n)] * eigen(Matrix(Δ_c), Matrix(M_c)).vectors[:,end-1]
v_f = F[:,Block.(1:n)] * eigen(Matrix(Δ_f), Matrix(M_f)).vectors[:,end-1]

g = range(-1,1; length=1000)
plot(g, v_c[g])
plot!(g, v_f[g])

eigen(Matrix(Δ_f), Matrix(M_f)).vectors[:,end]

filter(x -> abs(x) > 100eps(), eigvals(Matrix(R * Δ_c * R'), Matrix(R * M_c * R')))


eigvals(Matrix(R * Δ_c * pinv(R)), Matrix(R * M_c * pinv(R)))

filter(x -> abs(x) ≥ 100eps(), (eigvals(Matrix(-Δ_f + M_f) * R*inv(Matrix(-Δ_c + M_c)) * R')))