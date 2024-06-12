# PiecewiseOrthogonalPolynomials.jl
A Julia package for piecewise orthogonal polynomials which can be used in p-FEM


We can make piecewise integrated-Legendre bases using the quasi-matrix `ContinuousPolynomial{1}`:
```julia
using PiecewiseOrthogonalPolynomials, Plots

𝐗 = range(-1,1; length=4)
C = ContinuousPolynomial{1}(𝐗)
plot(C[:,Block(1)])
```
```julia
plot(C[:,Block.(2:3)])
```

The mass matrix can be constructed via:
```julia
M = C'C
```
We can also construct the stiffness matrix:
```julia
Δ = weaklaplacian(C)
```

We can truncate as follows:
```julia
N = 10
KR = Block.(Base.OneTo(N))
Mₙ = M[KR,KR]
Δₙ = Δ[KR,KR]
```

We can compute the reverse Cholesky in optimal complexity:
```julia
using MatrixFactorizations
L = reversecholesky(Symmetric(-Δₙ + Mₙ)).L
```