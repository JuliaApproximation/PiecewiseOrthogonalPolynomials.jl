using PiecewiseOrthogonalPolynomials, MatrixFactorizations, HypergeometricFunctions
using Elliptic
using ClassicalOrthogonalPolynomials, StaticArrays, LinearAlgebra
using Base: oneto

"""
Solve the Poisson equation with zero Dirichlet boundary conditions on the square
"""

# These 4 routines from ADI were lifted from Kars' M4R repo.
function mobius(z, a, b, c, d, α)
    t₁ = a*(-α*b + b + α*c + c) - 2b*c
    t₂ = a*(α*(b+c) - b + c) - 2α*b*c
    t₃ = 2a - (α+1)*b + (α-1)*c
    t₄ = -α*(-2a+b+c) - b + c

    (t₁*z + t₂)/(t₃*z + t₄)
end

ellipticK(z) = convert(eltype(α),π)/2*HypergeometricFunctions._₂F₁(one(α)/2,one(α)/2,1, z)


function ADI_shifts(J, a, b, c, d, tol=1e-15)
    γ = (c-a)*(d-b)/((c-b)*(d-a))
    α = -1 + 2γ + 2√Complex(γ^2 - γ)
    α = Real(α)

    K = ellipticK(1-1/big(α)^2)
    # K = Elliptic.K(1-1/α^2)
    dn = [Elliptic.Jacobi.dn((2*j + 1)*K/(2J), 1-1/α^2) for j = 0:J-1]

    [mobius(-α*i, a, b, c, d, α) for i = dn], [mobius(α*i, a, b, c, d, α) for i = dn]
end

function ADI(A, B, C, F, a, b, c, d, tol=1e-15)
    "ADI method for solving standard sylvester AX - XB = F"
    # Modified slightly by John to allow for the mass matrix
    n = size(A)[1]
    X = zeros(axes(A))

    γ = (c-a)*(d-b)/((c-b)*(d-a))
    J = Int(ceil(log(16γ)*log(4/tol)/π^2))
    # J = 200
    p, q = ADI_shifts(J, a, b, c, d, tol)

    for j = 1:J
        X = ((A/p[j] - C)*X - F/p[j])/reversecholesky(Symmetric(C - B/p[j]))
        X = reversecholesky(Symmetric(C - A/q[j]))\(X*(B/q[j] - C) - F/q[j])
    end

    X
end

function analysis_2D(f, n, p)
    dx = 2/n

    P₀ = legendre(0..dx)  # Legendre mapped to the reference cell
    z,T = ClassicalOrthogonalPolynomials.plan_grid_transform(P₀, (p, p))
    F = zeros(n*p, n*p)  # initialise F

    for i = 0:n-1  # loop over cells in positive x direction
        for j = 0:n-1  # loop over cells in positive y direction
            f_ = (x,y) -> f(x + i*dx - 1, y + j*dx - 1)  # define f on reference cell
            F[i+1:n:n*p, j+1:n:n*p] = T * f_.(z, z') # interpolate f into 2D tensor Legendre polynomials on reference cell
        end
    end

    F
end

r = range(-1, 1, 5)
K = length(r)-1

P = ContinuousPolynomial{0}(r)
Q = DirichletPolynomial(r)
Δ = -weaklaplacian(Q)
M = grammatrix(Q)

p = 40 # truncation degree on each cell
KR = Block.(oneto(p))
Δₙ = Δ[KR,KR]
Mₙ = M[KR,KR]

U = reversecholesky(Symmetric(Δₙ)).U

A = (U \ (U \ Mₙ)') # = L⁻¹ pΔ L⁻ᵀ
e1s, e2s = eigmin(A), eigmax(A)

z = SVector.(-1:0.01:1, (-1:0.01:1)')

# RHS
f = (x,y) -> -2 .*sin.(pi*x) .* (2pi*y .*cos.(pi*y) .+ (1-pi^2*y^2) .*sin.(pi*y))
fp = analysis_2D(f, K, p)  # interpolate F into P⊗P
Fa = P[first.(z)[:,1], KR] * fp  * P[first.(z)[:,1], KR]'
norm(splat(f).(z) - Fa)

# weak form for RHS
F = (Q'*P)[KR, KR]*fp*((Q'*P)[KR, KR])'  # RHS <f,v>

A, B, a, b, c, d = Mₙ, -Mₙ, e1s, e2s, -e2s, -e1s
@time X = ADI(A, B, Δₙ, F, a, b, c, d)

# X = UΔ
Y = (U' \ (U \ X'))'

u_exact = z -> ((x,y)= z; sin.(π*x)*sin.(π*y)*y^2)
Ua = Q[first.(z)[:,1], Block.(1:p)] * Y  * Q[first.(z)[:,1], Block.(1:p)]'

@test u_exact.(z) ≈ Ua # ℓ^∞ error.

"""
Via (5.3) and (5.6) of Kars' thesis.
"""
# Reverse Cholesky
rpM = pM[end:-1:1, end:-1:1]
L = cholesky(Symmetric(rpM)).L
L = L[end:-1:1, end:-1:1]
L * L' ≈ pM

# Compute spectrum
A = (L \ (L \ pΔ)') # = L⁻¹ pΔ L⁻ᵀ
e1s, e2s = eigmin(A), eigmax(A)

z = SVector.(-1:0.01:1, (-1:0.01:1)')

# RHS
f(z) = ((x,y)= z; -2 .*sin.(pi*x) .* (2pi*y .*cos.(pi*y) .+ (1-pi^2*y^2) .*sin.(pi*y)))
fp = analysis_2D(f, K, p)  # interpolate F into P⊗P
Fa = P[first.(z)[:,1], Block.(1:p)] * fp  * P[first.(z)[:,1], Block.(1:p)]'
norm(f.(z) - Fa)

# weak form for RHS
F = (C'*P)[Block.(1:p), Block.(1:p)]*fp*((C'*P)[Block.(1:p), Block.(1:p)])'  # RHS <f,v>
F[1, :] .= 0; F[K+1, :] .= 0; F[:, 1] .= 0; F[:, K+1] .= 0  # Dirichlet bcs

tol = 1e-15 # ADI tolerance
A, B, a, b, c, d = pΔ, -pΔ, e1s, e2s, -e2s, -e1s
X = ADI(A, B, pM, F, a, b, c, d, tol)

# X = UM
U = (L' \ (L \ X'))'

u_exact = z -> ((x,y)= z; sin.(π*x)*sin.(π*y)*y^2)
Ua = C[first.(z)[:,1], Block.(1:p)] * U  * C[first.(z)[:,1], Block.(1:p)]'

norm(u_exact.(z) - Ua) # ℓ^∞ error.

