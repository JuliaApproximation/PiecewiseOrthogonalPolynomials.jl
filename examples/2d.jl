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
M,N = Block(40),Block(41)
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
C = pl * F

# We approximate the original function to high-order
@test P[0.1,Block(1):M]' * C * P[0.2,Block(1):N] ≈ f(0.1,0.2)

# Further we can transform back:
@test pl \ C ≈ F


