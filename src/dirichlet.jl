struct DirichletPolynomial{T,P<:AbstractVector} <: AbstractPiecewisePolynomial{1,T,P}
    points::P
end

DirichletPolynomial{T}(pts::P) where {T,P} = DirichletPolynomial{T,P}(pts)
DirichletPolynomial(pts) = DirichletPolynomial{Float64}(pts)
DirichletPolynomial{T}(P::DirichletPolynomial) where {T} = DirichletPolynomial{T}(P.points)

ContinuousPolynomial{order}(P::DirichletPolynomial{T}) where {order, T} = ContinuousPolynomial{order}(P.points)
PiecewisePolynomial(P::DirichletPolynomial{T}) where {T} = PiecewisePolynomial(ContinuousPolynomial{0}(P))

axes(B::DirichletPolynomial) = (Inclusion(first(B.points) .. last(B.points)), blockedrange(Vcat(length(B.points)-2, Fill(length(B.points) - 1, ∞))))

show(io::IO, Q::DirichletPolynomial) = summary(io, Q)
summary(io::IO, Q::DirichletPolynomial) = print(io, "DirichletPolynomial($(Q.points))")


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
    BBBArrowheadMatrix(layout_getindex(Eye{T}(m), :, 2:m-1), (), (), Fill(Eye{T}(∞), m-1))
end

# \ understood as pseudo invers
function \(Q::DirichletPolynomial, C::ContinuousPolynomial{1})
    T = promote_type(eltype(Q), eltype(C))
    @assert C.points == Q.points
    m = length(C.points)
    BBBArrowheadMatrix(layout_getindex(Eye{T}(m), 2:m-1, :), (), (), Fill(Eye{T}(∞), m-1))
end

function \(P::ContinuousPolynomial, Q::DirichletPolynomial)
    C = ContinuousPolynomial{1}(Q)
    (P \ C) * (C \ Q)
end

###
# transforms
###


function adaptivetransform_ldiv(Q::DirichletPolynomial{V}, f::AbstractQuasiVector) where V
    C = ContinuousPolynomial{1}(Q)
    (Q \ C) * (C \ f)
end

struct DirichletPolynomialTransform{T, Pl<:Plan{T}, RRs, Dims} <: Plan{T}
    legendretransform::Pl
    R::RRs
    dims::Dims
end

DirichletPolynomialTransform(pl::ContinuousPolynomialTransform) =
    DirichletPolynomialTransform(pl.legendretransform, pl.R, pl.dims)

plan_transform(C::DirichletPolynomial{T}, Ns::NTuple{N,Block{1}}, dims=ntuple(identity,Val(N))) where {N,T} =
    DirichletPolynomialTransform(plan_transform(ContinuousPolynomial{1}(C), Ns, dims))

_tensorsize2dirichletblocks() = ()
_tensorsize2dirichletblocks(M,N, Ns...) = (Vcat(N-1, Fill(N, M-2)), _tensorsize2dirichletblocks(Ns...)...)
    
    
function *(Pl::DirichletPolynomialTransform{T,<:Any,<:Any,Int}, X::AbstractMatrix{T}) where T
    dat = Pl.R * (Pl.legendretransform*X)
    cfs = PseudoBlockArray{T}(undef,  _tensorsize2dirichletblocks(size(X)...)...)
    dims = Pl.dims
    @assert dims == 1

    M,N = size(X,1), size(X,2)
    if size(dat,1) ≥ 1
        for j = 1:N-1
            cfs[Block(1)[j]] = dat[2,j]
        end
    end
    cfs[Block.(2:M-1)] .= vec(dat[3:end,:]')
    cfs
end

function \(Pl::DirichletPolynomialTransform{T,<:Any,<:Any,Int}, cfs::AbstractVector{T}) where T
    dims = Pl.dims
    @assert dims == 1
    
    M,N = blocksize(cfs,1)+1, size(axes(cfs,1)[Block(1)],1)+1
    dat = Matrix{T}(undef, M, N)
    
    if M ≥ 1
        dat[1,1] = zero(T)
        for j = 1:N-1
            dat[2,j] = dat[1,j+1] = cfs[Block(1)[j]]
        end
        dat[2,end] = zero(T)
    end

    for j = 1:N, k = 3:M
        dat[k,j] = cfs[Block(k-1)[j]]
    end

    Pl.legendretransform \ (Pl.R \ dat)
end
    
    
    
function _dirichletpolyinds2blocks(k, j)
    k == 1 && return Block(1)[j-1]
    k == 2 && return Block(1)[j]
    Block(k-1)[j]
end
function *(Pl::DirichletPolynomialTransform{T}, X::AbstractArray{T,4}) where T
    dat = Pl.R * (Pl.legendretransform*X)
    cfs = PseudoBlockArray{T}(undef,  _tensorsize2dirichletblocks(size(X)...)...)
    dims = Pl.dims
    @assert dims == (1,2)

    M,N,O,P = size(X)
    for k = 1:M, j = 1:N, l = 1:O, m = 1:P
        k == j == 1 && continue
        k == 2 && j == N && continue
        l == m == 1 && continue
        l == 2 && m == P && continue
        cfs[_dirichletpolyinds2blocks(k,j), _dirichletpolyinds2blocks(l,m)] = dat[k,j,l,m]
    end
    cfs
end
    
function \(Pl::DirichletPolynomialTransform{T}, cfs::AbstractMatrix{T}) where T
    M,N = blocksize(cfs,1)+1, size(axes(cfs,1)[Block(1)],1)+1
    O,P = blocksize(cfs,2)+1, size(axes(cfs,2)[Block(1)],1)+1

    dat = Array{T}(undef,  M, N, O, P)
    dims = Pl.dims
    @assert dims == (1,2)

    for k = 1:M, j = 1:N, l = 1:O, m = 1:P
        if k == j == 1 ||
           (k == 2 && j == N) ||
            l == m == 1 ||
           (l == 2 && m == P)
           dat[k,j,l,m] = zero(T)
        else
            dat[k,j,l,m] = cfs[_dirichletpolyinds2blocks(k,j), _dirichletpolyinds2blocks(l,m)]
        end
    end

    Pl.legendretransform \ (Pl.R \ dat)
end
    
###
# weaklaplacian
###    

function weaklaplacian(C::DirichletPolynomial{T,<:AbstractRange}) where T
    r = C.points
    N = length(r)
    s = step(r)
    si = inv(s)
    t1 = Fill(-2si, N-2)
    t2 = Fill(si, N-3)
    Symmetric(BBBArrowheadMatrix(LazyBandedMatrices.Bidiagonal(t1, t2, :U), (), (),
        Fill(Diagonal(convert(T, -4) ./ (s*(convert(T,3):2:∞))), N-1)))
end

function grammatrix(Q::DirichletPolynomial{T, <:AbstractRange}) where T
    r = Q.points

    N = length(r) - 1
    h = step(r) # 2/N
    a = [-2 /((2k+1)*(2k+3)*(2k+5)) for k = -1:∞]
    b = [4 /((2k+1)*(2k+3)*(2k+5)) for k = 0:∞]

    a11 = LazyBandedMatrices.Bidiagonal(Fill(2h/3, N-1), Fill(h/6, N-2), :U)
    a21 = _BandedMatrix(Fill(h/6, 2, N), N-1, 0, 1)
    a31 = _BandedMatrix(Vcat(Fill(-h/30, 1, N), Fill(h/30, 1, N)), N-1, 0, 1)

    Symmetric(BBBArrowheadMatrix(a11, (a21, a31), (),
                Fill(_BandedMatrix(Vcat((h*a/2)',
                Zeros{T}(1,∞),
                (h*b/2)'), ∞, 0, 2), N)))
end

function grammatrix(Q::DirichletPolynomial)
    C = ContinuousPolynomial{1}(Q)
    R = C \ Q
    R' * grammatrix(C) * R
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

for grd in (:grid, :plotgrid)
    @eval begin
        $grd(C::DirichletPolynomial, n::Integer) = $grd(ContinuousPolynomial{1}(C.points), n)
        $grd(C::DirichletPolynomial, n::Block{1}) = $grd(ContinuousPolynomial{1}(C.points), n)
    end
end

###
# singularities
###

singularities(C::DirichletPolynomial) = C
basis_singularities(C::DirichletPolynomial) = C
singularitiesbroadcast(_, Q::DirichletPolynomial) = ContinuousPolynomial{1}(Q) # Assume we stay smooth but might not vanish
singularitiesbroadcast(::typeof(sin), Q::DirichletPolynomial) = Q # Smooth functions such that f(0) == 0 preserve behaviour