using PiecewiseOrthogonalPolynomials, Plots

## Goal 1: Do hp solves in 1D where complexity is optimal 
# regardless of h or p

r = range(0, 1; length=6)

# C is standard affine FEM combined with mapped (1-x^2) * P_k^(1,1)(x)
C = ContinuousPolynomial{1}(r)

g = range(0,1; length=1000)
plot(g, C[g,Block(4)])

x = axes(C,1)
D = Derivative(x)

M = C'C # Mass matrix
Δ = -(D*C)'*(D*C) # Weak Laplacian


# For 2D/3D use Fortanto & Townsend have fast spectral Poisson/Helmholtz that
# only uses a fast 1D solve + assumptions on nice spectrum using ADI solver