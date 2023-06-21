using PiecewiseOrthogonalPolynomials

P = ContinuousPolynomial{1}(range(-1,1; length=4))
M = P'P

