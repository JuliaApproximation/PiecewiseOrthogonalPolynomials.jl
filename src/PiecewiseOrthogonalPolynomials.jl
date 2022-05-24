module PiecewiseOrthogonalPolynomials
using ClassicalOrthogonalPolynomials, LinearAlgebra, BlockArrays, BlockBandedMatrices, ContinuumArrays, QuasiArrays, LazyArrays, LazyBandedMatrices, FillArrays

import BlockArrays: BlockSlice, block, blockindex, blockvec
import BlockBandedMatrices: _BandedBlockBandedMatrix
import ClassicalOrthogonalPolynomials: grid, massmatrix
import ContinuumArrays: @simplify, factorize, TransformFactorization, AbstractBasisLayout, MemoryLayout, layout_broadcasted, ExpansionLayout, basis
import LazyArrays: paddeddata
import LazyBandedMatrices: BlockBroadcastMatrix
import Base: axes, getindex, ==, \, OneTo

export PiecewisePolynomial, ContinuousPolynomial, Derivative, Block

abstract type AbstractPiecewisePolynomial{order,T,P<:AbstractVector} <: Basis{T} end

struct PiecewisePolynomialLayout{order} <: AbstractBasisLayout end
MemoryLayout(::Type{<:AbstractPiecewisePolynomial{order}}) where order = PiecewisePolynomialLayout{order}()

struct PiecewisePolynomial{T,Bas,P<:AbstractVector} <: AbstractPiecewisePolynomial{0,T,P}
    basis::Bas
    points::P
end

PiecewisePolynomial{T}(basis::AbstractQuasiMatrix, points::AbstractVector) where {T} =
    PiecewisePolynomial{T,typeof(basis),typeof(points)}(basis, points)
PiecewisePolynomial(basis::AbstractQuasiMatrix{T}, points::AbstractVector) where {T} =
    PiecewisePolynomial{T}(basis, points)

axes(B::PiecewisePolynomial) = (Inclusion(first(B.points) .. last(B.points)), blockedrange(Fill(length(B.points) - 1, ∞)))

function repeatgrid(ax, g, pts)
    ret = Matrix{eltype(g)}(undef, length(g), length(pts) - 1)
    for j in axes(ret, 2)
        ret[:, j] = affine(ax, pts[j] .. pts[j+1])[g]
    end
    ret
end

function grid(V::SubQuasiArray{T,2,<:PiecewisePolynomial,<:Tuple{Inclusion,BlockSlice}}) where {T}
    P = parent(V)
    _, JR = parentindices(V)
    N = Int(last(JR))
    g = grid(P.basis[:, OneTo(N)])
    repeatgrid(axes(P.basis, 1), g, P.points)
end

struct ApplyFactorization{T, FF, FAC<:Factorization{T}} <: Factorization{T}
    f::FF
    F::FAC
end

\(P::ApplyFactorization, f) = P.f(P.F \ f)


function factorize(V::SubQuasiArray{<:Any,2,<:PiecewisePolynomial,<:Tuple{Inclusion,BlockSlice}})
    P = parent(V)
    _,JR = parentindices(V)
    N = Int(last(JR.block))
    F = factorize(view(P.basis,:,OneTo(N)), length(P.points)-1)
    ApplyFactorization(v -> blockvec(permutedims(v)), TransformFactorization(repeatgrid(axes(P.basis, 1), F.grid, P.points), F.plan))
end


struct ContinuousPolynomial{order,T,P<:AbstractVector} <: AbstractPiecewisePolynomial{order,T,P}
    points::P
end


ContinuousPolynomial{o,T}(pts::P) where {o,T,P} = ContinuousPolynomial{o,T,P}(pts)
ContinuousPolynomial{o}(pts) where {o} = ContinuousPolynomial{o,Float64}(pts)
ContinuousPolynomial{o}(P::ContinuousPolynomial) where {o} = ContinuousPolynomial{o,eltype(P)}(P.points)

PiecewisePolynomial(P::ContinuousPolynomial{0,T}) where {T} = PiecewisePolynomial(Legendre{T}(), P.points)

axes(B::ContinuousPolynomial{0}) where {o} = axes(PiecewisePolynomial(B))
axes(B::ContinuousPolynomial{1}) where {o} =
    (Inclusion(first(B.points) .. last(B.points)), blockedrange(Vcat(length(B.points), Fill(length(B.points) - 1, ∞))))

==(::PiecewisePolynomial, ::ContinuousPolynomial{1}) = false
==(::ContinuousPolynomial{1}, ::PiecewisePolynomial) = false
==(A::ContinuousPolynomial{o}, B::ContinuousPolynomial{o}) where o = A.points == B.points

function getindex(P::PiecewisePolynomial{T}, x::Number, Kk::BlockIndex{1}) where {T}
    K, k = block(Kk), blockindex(Kk)
    b = searchsortedlast(P.points, x)
    if b == length(P.points) == k + 1 # last point
        P.basis[affine(P.points[end-1] .. P.points[end], axes(P.basis, 1))[x], Int(K)]
    elseif b == k
        P.basis[affine(P.points[b] .. P.points[b+1], axes(P.basis, 1))[x], Int(K)]
    else
        zero(T)
    end
end
getindex(P::ContinuousPolynomial{0,T}, x::Number, Kk::BlockIndex{1}) where {T} = PiecewisePolynomial(P)[x, Kk]

function getindex(P::ContinuousPolynomial{1,T}, x::Number, Kk::BlockIndex{1}) where {T}
    K, k = block(Kk), blockindex(Kk)
    if K == Block(1)
        LinearSpline(P.points)[x, k]
    else
        b = searchsortedlast(P.points, x)
        if b == k
            Weighted(Jacobi{T}(1, 1))[affine(P.points[b] .. P.points[b+1], ChebyshevInterval{real(T)}())[x], Int(K)-1]
        else
            zero(T)
        end
    end
end


getindex(P::AbstractPiecewisePolynomial, x::Number, k::Int) = P[x, findblockindex(axes(P, 2), k)]

factorize(V::SubQuasiArray{T,2,<:ContinuousPolynomial{0},<:Tuple{Inclusion,BlockSlice}}) where {T} =
    factorize(view(PiecewisePolynomial(parent(V)), parentindices(V)...))
grid(V::SubQuasiArray{T,2,<:ContinuousPolynomial{0},<:Tuple{Inclusion,BlockSlice}}) where {T} =
    grid(view(PiecewisePolynomial(parent(V)), parentindices(V)...))

function grid(V::SubQuasiArray{T,2,<:ContinuousPolynomial{1},<:Tuple{Inclusion,BlockSlice}}) where {T}
    P = parent(V)
    _, JR = parentindices(Q)
    pts = P.points
    grid(view(PiecewisePolynomial(Weighted(Jacobi{T}(1, 1)), pts), :, JR))
end

#######
# Conversion
#######

function \(P::ContinuousPolynomial{0}, C::ContinuousPolynomial{1})
    T = promote_type(eltype(P), eltype(C))
    # diag blocks based on
    # L = Legendre{T}() \ Weighted(Jacobi{T}(1,1))
    @assert P.points == C.points
    N = length(P.points)
    v = mortar(Fill.((convert(T, 2):2:∞) ./ (3:2:∞), N - 1))
    z = Zeros{T}(axes(v))
    H1 = BlockBroadcastArray(hcat, z, v)
    M1 = BlockVcat(Zeros{T}(N, 2), H1)
    M2 = BlockVcat(Ones{T}(N, 2) / 2, Zeros{T}((axes(v, 1), Base.OneTo(2))))
    H3 = BlockBroadcastArray(hcat, z, -v)
    M3 = BlockVcat(Hcat(Ones{T}(N) / 2, -Ones{T}(N) / 2), H3)
    dat = BlockHcat(M1, M2, M3)'
    _BandedBlockBandedMatrix(dat, axes(P, 2), (1, 1), (0, 1))
end

######
# Gram matrix
######

@simplify function *(Ac::QuasiAdjoint{<:Any,<:ContinuousPolynomial{0}}, B::ContinuousPolynomial{0})
    A = Ac'
    T = promote_type(eltype(A), eltype(B))
    r = A.points
    @assert r == B.points
    N = length(r)
    M = massmatrix(Legendre{T}())
    Diagonal(mortar(Fill.((step(r) / 2) .* M.diag, N - 1)))
end


@simplify function *(Ac::QuasiAdjoint{<:Any,<:ContinuousPolynomial}, B::ContinuousPolynomial)
    A = Ac'
    P = ContinuousPolynomial{0}(A)
    Q = ContinuousPolynomial{0}(B)
    (P \ A)' * (P'Q) * (Q \ B)
end

#####
# Derivative
#####

@simplify function *(D::Derivative, C::ContinuousPolynomial{1})
    T = promote_type(eltype(D), eltype(C))

    # Legendre() \ (D*Weighted(Jacobi(1,1)))
    r = C.points
    N = length(r)
    v = mortar(Fill.((-convert(T, 2):-2:-∞), N - 1))
    z = Zeros{T}(axes(v))
    H = BlockBroadcastArray(hcat, z, v)
    M = BlockVcat(Hcat(Ones{T}(N) / 2, -Ones{T}(N) / 2), H)
    P = ContinuousPolynomial{0}(C)
    P * _BandedBlockBandedMatrix(M', (axes(P, 2), axes(C, 2)), (0, 0), (0, 1))
end


### multiplication

function layout_broadcasted(::Tuple{ExpansionLayout{PiecewisePolynomialLayout{0}},PiecewisePolynomialLayout{0}}, ::typeof(*), a, P)
    @assert basis(a) == P
    _,c = a.args
    T = eltype(c)
    m = length(P.points)
    B = PiecewisePolynomial(P).basis

    ops = [(cₖ = [paddeddata(c)[k:m-1:end]; Zeros{T}(∞)];
            aₖ = B  * cₖ;
            unitblocks(B \ (aₖ .* B))) for k = 1:m-1]

    P * BlockBroadcastMatrix{T}(Diagonal, ops...)
end


function layout_broadcasted(::Tuple{ExpansionLayout{PiecewisePolynomialLayout{0}},PiecewisePolynomialLayout{1}}, ::typeof(*), a, C)
    P = ContinuousPolynomial{0}(C)
    (a .* P) * (P \ C)
end


end # module
