struct DirichletPolynomial{T,P<:AbstractVector} <: AbstractPiecewisePolynomial{order,T,P}
    points::P
end

DirichletPolynomial{T}(pts::P) where {T,P} = DirichletPolynomial{T,P}(pts)
DirichletPolynomial(pts) = DirichletPolynomial{Float64}(pts)
DirichletPolynomial{T}(P::DirichletPolynomial) where {T} = DirichletPolynomial{T}(P.points)
DirichletPolynomial(P::DirichletPolynomial) = DirichletPolynomial{eltype(P)}(P)

ContinuousPolynomial{order}(P::DirichletPolynomial{T}) where {order, T} = ContinuousPolynomial{order}(P.points)
PiecewisePolynomial(P::DirichletPolynomial{T}) where {T} = PiecewisePolynomial(ContinuousPolynomial{0}(P))

axes(B::DirichletPolynomial{1}) =
    (Inclusion(first(B.points) .. last(B.points)), blockedrange(Vcat(length(B.points)-2, Fill(length(B.points) - 1, ∞))))

==(P::PiecewisePolynomial, C::DirichletPolynomial) = false
==(C::DirichletPolynomial, P::PiecewisePolynomial) = false
==(P::ContinuousPolynomial, C::DirichletPolynomial) = false
==(C::DirichletPolynomial, P::ContinuousPolynomial) = false


function getindex(P::DirichletPolynomial{T}, x::Number, Kk::BlockIndex{1}) where {T}
    K, k = block(Kk), blockindex(Kk)
    K == Block(1) && return ContinuousPolynomial{1}(P)[x, Block(1)[k+1]]
    ContinuousPolynomial{1}(P)[x, Kk]
end

# Conversion

function \(C::ContinuousPolynomial{1}, D::DirichletPolynomial)
    T = promote_type(eltype(P), eltype(C))
    @assert C.points == D.points
    m = length(C.points)
    ArrowheadMatrix(layout_getindex(Eye{T}(m), :, 2:m-1), (), (), Fill(Eye{T}(∞), m-1))
end