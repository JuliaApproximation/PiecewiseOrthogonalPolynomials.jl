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

==(P::PiecewisePolynomial, Q::PiecewisePolynomial) = P.basis == Q.basis && P.points == Q.points

function repeatgrid(ax, g, pts)
    ret = Matrix{eltype(g)}(undef, length(g), length(pts) - 1)
    for j in axes(ret, 2)
        ret[:, j] = affine(ax, pts[j] .. pts[j+1])[g]
    end
    ret
end


function grid(P::PiecewisePolynomial, N::Block{1})
    g = grid(P.basis, Int(N))
    repeatgrid(axes(P.basis, 1), g, P.points)
end

function plotgrid(P::PiecewisePolynomial, N::Block{1})
    g = plotgrid(P.basis, Int(N))
    vec(repeatgrid(axes(P.basis, 1), g, P.points)[end:-1:1,:]) # sort
end



grid(P::PiecewisePolynomial, n::Int) = grid(P, findblock(axes(P,2),n))
plotgrid(P::PiecewisePolynomial, n::Int) = plotgrid(P, findblock(axes(P,2),n))




struct ApplyFactorization{T, FF, FAC<:Factorization{T}} <: Factorization{T}
    f::FF
    F::FAC
end

\(P::ApplyFactorization, f) = P.f(P.F \ f)


_perm_blockvec(X::AbstractMatrix) = BlockVec(transpose(X))
function _perm_blockvec(X::AbstractArray{T,3}) where T
    X1 = _perm_blockvec(X[:,:,1])
    ret = PseudoBlockMatrix{T}(undef, (axes(X1,1), axes(X,3)))
    ret[:,1] = X1
    for k = 2:size(X,3)
        ret[:,k] = _perm_blockvec(X[:,:,k])
    end
    ret
end

function factorize(V::SubQuasiArray{<:Any,2,<:PiecewisePolynomial,<:Tuple{Inclusion,BlockSlice}}, dims...)
    P = parent(V)
    _,JR = parentindices(V)
    N = Int(last(JR.block))
    x,F = plan_grid_transform(P.basis, Array{eltype(P)}(undef, N, length(P.points)-1, dims...), 1)
    ApplyFactorization(_perm_blockvec, TransformFactorization(repeatgrid(axes(P.basis, 1), x, P.points), F))
end


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

getindex(P::AbstractPiecewisePolynomial, x::Number, k::Int) = P[x, findblockindex(axes(P, 2), k)]


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
