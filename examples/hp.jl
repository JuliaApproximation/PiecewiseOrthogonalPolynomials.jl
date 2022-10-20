using PiecewiseOrthogonalPolynomials, Plots

## Goal 1: Do hp solves in 1D where complexity is optimal 
# regardless of h or p

r = range(0, 1; length=4)

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

using ClassicalOrthogonalPolynomials

W = Weighted(Jacobi(1,1))
x = axes(W,1)
D = Derivative(x)
P = Legendre()

# Strong form
Δ = Jacobi(1,1) \ D^2 * W # secibd derivat
M = Jacobi(1,1) \ W # conversion

# Weak form

Δ = -(D*W)'*(D*W)
M = W'W

V = P / P \ x.^2; V = W'*P * (P \ (V .* P)) * (P \ W)

Δ  + V

r = range(0, 1; length=3)

# C is standard affine FEM combined with mapped (1-x^2) * P_k^(1,1)(x)
C = ContinuousPolynomial{1}(r)




W[0.1,5]
g = range(0,1 ;length=100)
plot(g, C[g,6])
f = W * [1; 2; 3; zeros(∞)]

C'C
D = 


f'f

f[0.1]

g = range(-1,1; length=100)
plot(g, W[g,1:5])



p = 20; A = Matrix((Δ + M)[Block.(1:p+1), Block.(1:p+1)])

L = cholesky(Symmetric(A[end:-1:1, end:-1:1])).U[end:-1:1,end:-1:1]
@test L'*L ≈ A
spy(L)

using ClassicalOrthogonalPolynomials

W = Weighted(Jacobi(1,1))
n = 10_000; @time eigvals(Symmetric((W'W)[1:n,1:n]))

n = 10_000; @time eigen(Symmetric((W'W)[1:n,1:n]));

@time eigen(Symmetric(rand(10_000,10_000)));
