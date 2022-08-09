using PiecewiseOrthogonalPolynomials, Plots

p = reverse(2.0 .^ (-(0:10)))

C = ContinuousPolynomial{1}(p)
x = axes(C,1)
a = C / C \ log.(x)


a = C[:,1]

C \ (a .* C[:,1])