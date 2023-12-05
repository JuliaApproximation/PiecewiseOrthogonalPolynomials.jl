###
# This example shows how we can expand functions on a 2d rectangle using plan_transform. 
# This is a "manual mode" interface that might in the future be cleaned up to a user-facing
# interface.
###


using PiecewiseOrthogonalPolynomials, CairoMakie, Test
using PiecewiseOrthogonalPolynomials: plan_grid_transform

r = range(-1,1; length=3)
P = ContinuousPolynomial{0}(r)

# degrees in each dimension
M,N = Block(80),Block(81)
((x,y),pl) = plan_grid_transform(P, (M,N))



# x and y are matrices of points in each dimension, with the columns corresponding
# to different elements . This is needed to leverage
# fast 1D transforms acting on arrays.  Thus to transform we make a tensor grid as
# a 4-tensor. The way to do this is as follows:

f = (x,y) -> exp(x*cos(10y))
F = f.(x, reshape(y, 1, 1, size(y,1),:))

# F contains our function samples. to plot we need to reduce back to vector/matrices.
# But an issue is that the sample points are in the reverse order. Thus we need to do
# some reordering:
surface(vec(x[end:-1:1,:]), vec(y[end:-1:1,:]), reshape(F[end:-1:1,:,end:-1:1,:], length(x), length(y)))


# We can now transform the samples to coefficient space:
X = pl * F

# We approximate the original function to high-order
@test P[0.1,Block(1):M]' * X * P[0.2,Block(1):N] ≈ f(0.1,0.2)

# Further we can transform back:
@test pl \ X ≈ F


# If we use ContinuousPolynomial{1} we can differentiate. This can be done as follows:

C = ContinuousPolynomial{1}(r)

# Compute coefficients of f in tensor product of C^(1)

(x_C,y_C),pl_C = plan_grid_transform(C, (M,N))
F_C = f.(x_C, reshape(y_C, 1, 1, size(y_C,1),:))
X_C = pl_C * F_C

@test C[0.1,Block(1):M]' * X_C * C[0.2,Block(1):N] ≈ f(0.1,0.2)

# Make the 1D differentiation matrices from C to P:

D_x = (P \ diff(C))[Block(1):M, Block(1):M]
D_y = (P \ diff(C))[Block(1):N, Block(1):N]


# We now compare this to an analytical derivative at a poiunt

f_xy = (x,y) -> -10sin(10y)*exp(x*cos(10y)) - 10cos(10y)*x*sin(10y)*exp(x*cos(10y))
@test P[0.1,Block(1):M]'*(D_x*X_C*D_y')*P[0.2,Block(1):N] ≈ f_xy(0.1,0.2)

# We can transform back to get the values on a large grid:

F_xy = pl \ (D_x*X_C*D_y')

@test F_xy ≈ f_xy.(x, reshape(y,1,1,size(y)...))

surface(vec(x[end:-1:1,:]), vec(y[end:-1:1,:]), reshape(F_xy[end:-1:1,:,end:-1:1,:], length(x), length(y)))

