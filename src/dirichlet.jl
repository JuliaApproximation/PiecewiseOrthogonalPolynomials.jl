struct DirichletPolynomial{T,P<:AbstractVector} <: AbstractPiecewisePolynomial{1,T,P}
    points::P
end

DirichletPolynomial{T}(pts::P) where {T,P} = DirichletPolynomial{T,P}(pts)
DirichletPolynomial(pts) = DirichletPolynomial{Float64}(pts)
DirichletPolynomial{T}(P::DirichletPolynomial) where {T} = DirichletPolynomial{T}(P.points)

ContinuousPolynomial{order}(P::DirichletPolynomial{T}) where {order, T} = ContinuousPolynomial{order}(P.points)
PiecewisePolynomial(P::DirichletPolynomial{T}) where {T} = PiecewisePolynomial(ContinuousPolynomial{0}(P))

axes(B::DirichletPolynomial) = (Inclusion(first(B.points) .. last(B.points)), blockedrange(Vcat(length(B.points)-2, Fill(length(B.points) - 1, ∞))))

==(P::DirichletPolynomial, C::DirichletPolynomial) = P.points == C.points
==(P::PiecewisePolynomial, C::DirichletPolynomial) = false
==(C::DirichletPolynomial, P::PiecewisePolynomial) = false
==(P::ContinuousPolynomial, C::DirichletPolynomial) = false
==(C::DirichletPolynomial, P::ContinuousPolynomial) = false


function getindex(P::DirichletPolynomial{T}, x::Number, Kk::BlockIndex{1}) where {T}
    K, k = block(Kk), blockindex(Kk)
    K == Block(1) && return ContinuousPolynomial{1}(P)[x, Block(1)[k+1]]
    ContinuousPolynomial{1}(P)[x, Kk]
end

#####
# Conversion
#####

function \(C::ContinuousPolynomial{1}, Q::DirichletPolynomial)
    T = promote_type(eltype(Q), eltype(C))
    @assert C.points == Q.points
    m = length(C.points)
    ArrowheadMatrix(layout_getindex(Eye{T}(m), :, 2:m-1), (), (), Fill(Eye{T}(∞), m-1))
end

# \ understood as pseudo invers
function \(Q::DirichletPolynomial, C::ContinuousPolynomial{1})
    T = promote_type(eltype(Q), eltype(C))
    @assert C.points == Q.points
    m = length(C.points)
    ArrowheadMatrix(layout_getindex(Eye{T}(m), 2:m-1, :), (), (), Fill(Eye{T}(∞), m-1))
end


function adaptivetransform_ldiv(Q::DirichletPolynomial{V}, f::AbstractQuasiVector) where V
    C = ContinuousPolynomial{1}(Q)
    (Q \ C) * (C \ f)
end

function weaklaplacian(C::DirichletPolynomial{T,<:AbstractRange}) where T
    r = C.points
    N = length(r)
    s = step(r)
    si = inv(s)
    t1 = Fill(-2si, N-2)
    t2 = Fill(si, N-3)
    Symmetric(ArrowheadMatrix(LazyBandedMatrices.Bidiagonal(t1, t2, :U), (), (),
        Fill(Diagonal(convert(T, -16) .* (1:∞) .^ 2 ./ (s .* ((2:2:∞) .+ 1))), N-1)))
end

function grammatrix(Q::DirichletPolynomial{T, <:AbstractRange}) where T
    r = Q.points

    N = length(r) - 1
    h = step(r) # 2/N
    a = ((convert(T,4):4:∞) .* (convert(T,-2):2:∞)) ./ ((1:2:∞) .* (3:2:∞) .* (-1:2:∞))
    b = (((convert(T,2):2:∞) ./ (3:2:∞)).^2 .* (convert(T,2) ./ (1:2:∞) .+ convert(T,2) ./ (5:2:∞)))

    a11 = LazyBandedMatrices.Bidiagonal(Fill(2h/3, N-1), Fill(h/6, N-2), :U)
    a21 = _BandedMatrix(Fill(h/3, 2, N), N-1, 0, 1)
    a31 = _BandedMatrix(Vcat(Fill(-2h/15, 1, N), Fill(2h/15, 1, N)), N-1, 0, 1)

    Symmetric(ArrowheadMatrix(a11, (a21, a31), (),
                Fill(_BandedMatrix(Vcat((-h*a/2)',
                Zeros{T}(1,∞),
                (h*b/2)'), ∞, 0, 2), N)))
end

@simplify function *(Ac::QuasiAdjoint{<:Any,<:DirichletPolynomial}, B::ContinuousPolynomial)
    A = Ac'
    C = ContinuousPolynomial{1}(A)
    (A\C) * (C'B)
end

function diff(Q::DirichletPolynomial{T}; dims=1) where T
    C = ContinuousPolynomial{1}(Q)
    diff(C; dims=dims) * (C \ Q)
end