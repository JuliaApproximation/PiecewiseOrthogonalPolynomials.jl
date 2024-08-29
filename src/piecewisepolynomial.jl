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




struct ApplyPlan{T, FF, FAC<:Plan{T}, Args} <: Plan{T}
    f::FF
    F::FAC
    args::Args
end

*(P::ApplyPlan, f::AbstractArray) = P.f(P.F * f, P.args...)

size(P::ApplyPlan, k...) = size(P.F, k...)

"""
    _perm_blockvec

takes a matrix and constructs a blocked-vector where the different
rows correspond to different blocks.
"""
function _perm_blockvec(X::AbstractMatrix, dims=1)
    @assert dims == 1 || dims == (1,) || dims == 1:1
    BlockVec(transpose(X))
end

"""
    _inv_perm_blockvec

is the inverse of _perm_blockvec
"""
function _inv_perm_blockvec(X::ApplyVector{<:Any,typeof(blockvec)}, dims = 1)
    @assert dims == 1 || dims == (1,) || dims == 1:1
    transpose(only(X.args))
end

function _perm_blockvec(X::AbstractArray{T,3}, dims=1) where T
    @assert dims == 1
    X1 = _perm_blockvec(X[:,:,1])
    ret = BlockedMatrix{T}(undef, (axes(X1,1), axes(X,3)))
    ret[:,1] = X1
    for k = 2:size(X,3)
        ret[:,k] = _perm_blockvec(X[:,:,k])
    end
    ret
end


function _perm_blockvec(X::AbstractArray{T,4}, dims=(1,2)) where T
    @assert dims == 1:2 || dims == (1,2)
    X1 = _perm_blockvec(X[:,:,1,1])
    X2 = _perm_blockvec(X[1,1,:,:])
    ret = BlockedMatrix{T}(undef, (axes(X1,1), axes(X2,1)))
    for k = axes(X,1), j = axes(X,2), l = axes(X,3), m = axes(X,4)
        ret[Block(k)[j], Block(l)[m]] = X[k,j,l,m]
    end
    ret
end

function _inv_perm_blockvec(X::AbstractMatrix{T}, dims=(1,2)) where T
    @assert dims == 1:2 || dims == (1,2) || dims == 1
    M,N = blocksize(X)
    m,n = size(X)
    if dims == 1:2 || dims == (1,2)
        ret = Array{T}(undef, M, m ÷ M, N, n ÷ N)
        for k = axes(ret,1), j = axes(ret,2), l = axes(ret,3), m = axes(ret,4)
            ret[k,j,l,m] = X[Block(k)[j], Block(l)[m]]
        end
    elseif dims == 1
        ret = Array{T}(undef, M, m ÷ M, n ÷ N)
        for k = axes(ret,1), j = axes(ret,2), l = axes(ret,3)
            ret[k,j,l] = X[Block(k)[j], Block(1)[l]]
        end
    end
    ret
end

function _perm_blockvec(X::AbstractArray{T,5}, dims=(1,2)) where T
    @assert dims == 1:2 || dims == (1,2)
    X1 = _perm_blockvec(X[:,:,:,:,1])
    ret = BlockedArray{T}(undef, (axes(X1,1), axes(X1,2), 1:2))
    ret[:, :, 1] = X1
    for k = 2:lastindex(ret,3)
        ret[:, :, k] = _perm_blockvec(X[:,:,:,:,k])
    end
    ret
end

function _inv_perm_blockvec(X::AbstractArray{T,3}, dims=(1,2)) where T
    @assert dims == 1:2 || dims == (1,2)
    M,N,L = blocksize(X)
    m,n,ℓ = size(X)

    ret = Array{T}(undef, M, m ÷ M, N, n ÷ N, ℓ÷L)
    for k = axes(ret,5)
        ret[:,:,:,:,k] = _inv_perm_blockvec(X[:,:,k])
    end
    ret
end

\(F::ApplyPlan{<:Any,typeof(_perm_blockvec)}, X::AbstractArray) = F.F \ _inv_perm_blockvec(X, F.args...)

_interlace_const(n) = ()
_interlace_const(n, m, ms...) = (m, n, _interlace_const(n, ms...)...)

_doubledims(d::Int) = 2d-1
_doubledims(d::Int, e...) = tuple(_doubledims(d), _doubledims(e...)...)


# we transform a piecewise transform into a tensor transform where each even dimensional slice corresponds to a different piece.
# that is, we don't transform the last dimension.
function plan_transform(P::PiecewisePolynomial, Ns::NTuple{N,Block{1}}, dims=ntuple(identity,Val(N))) where N
    @assert dims == 1:N || dims == ntuple(identity,Val(N)) || (N == dims == 1)
    F = plan_transform(P.basis, _interlace_const(length(P.points)-1, Int.(Ns)...), _doubledims(dims...))
    ApplyPlan(_perm_blockvec, F,  (dims,))
end


# If one dimension is an integer then this means its a vector transform. That is, we are only transforming
# along one dimension.We add an extra dimension for the different entries in the vectors.
function plan_transform(P::PiecewisePolynomial, (M,n)::Tuple{Block{1},Int}, dims::Int)
    @assert dims == 1
    F = plan_transform(P.basis, (Int(M), length(P.points)-1, n), dims)
    ApplyPlan(_perm_blockvec, F, (dims,))
end

function plan_transform(P::PiecewisePolynomial, (N,M,n)::Tuple{Block{1},Block{1},Int}, dims=ntuple(identity,Val(2)))
    @assert dims == 1:2 || dims == ntuple(identity,Val(2))
    Ns = (N,M)
    F = plan_transform(P.basis, (_interlace_const(length(P.points)-1, Int.(Ns)...)..., n), _doubledims(dims...))
    ApplyPlan(_perm_blockvec, F, (dims,))
end

function factorize(V::SubQuasiArray{<:Any,2,<:PiecewisePolynomial,<:Tuple{Inclusion,BlockSlice}}, dims...)
    P = parent(V)
    _,JR = parentindices(V)
    TransformFactorization(plan_grid_transform(P, (last(JR.block), dims...), 1)...)
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


###
# singularities
###

singularities(C::PiecewisePolynomial) = C
basis_singularities(C::PiecewisePolynomial) = C
singularitiesbroadcast(_, C::PiecewisePolynomial) = C # Assume we stay piecewise smooth

####
# sum
####
_sum(P::PiecewisePolynomial, dims) = blockvec(diff(P.points)/2 .* sum(P.basis; dims=1))'