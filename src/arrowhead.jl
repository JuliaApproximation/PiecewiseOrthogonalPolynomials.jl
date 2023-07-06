
"""
LeftArrowheadMatrix

    B   D   D  
    B   D   D  D
    …   …   …   …   …
    B   D   D  
"""
struct LeftArrowheadMatrix{T} <: AbstractBandedBlockBandedMatrix{T}
    firstcol::Vector{BandedMatrix{T}}
    tail::Vector{BandedMatrix{T}}
end

subblockbandwidths(A::LeftArrowheadMatrix) = bandwidths(A.firstcol[1])
function blockbandwidths(A::LeftArrowheadMatrix)
    l,u = bandwidths(A.tail[1])
    max(l,length(A.firstcol))-1,u+1
end

function axes(L::LeftArrowheadMatrix)
    ξ,n = size(L.firstcol[1])
    m = length(L.tail)
    μ,ν = size(L.tail[1])
    blockedrange(Vcat(ξ, Fill(m,μ-1))), blockedrange(Vcat(n, Fill(m,ν)))
end

function getindex(L::LeftArrowheadMatrix{T}, Kk::BlockIndex{1}, Jj::BlockIndex{1}) where T
    K,k = block(Kk),blockindex(Kk)
    J,j = block(Jj),blockindex(Jj)
    J == Block(1) && return L.firstcol[Int(K)][k,j]
    k ≠ j && return zero(T)
    L.tail[k][Int(K), Int(J)-1]
end

function getindex(L::LeftArrowheadMatrix, k::Int, j::Int)
    ax,bx = axes(L)
    L[findblockindex(ax, k), findblockindex(bx, j)]
end

function LeftArrowheadMatrix(firstcol, tail)
    n = size(firstcol[1],2)
    m = length(tail)
    λ,μ = bandwidths(firstcol[1])
    l,u = bandwidths(tail[1])
    for op in firstcol[2:end]
        @assert size(op) == (m,n)
        λₖ,μₖ = bandwidths(op)
        @assert λₖ ≤ λ
        @assert μₖ ≤ μ
    end
    for op in tail
        @assert bandwidths(op) == (l,u)
    end
    T = promote_type(eltype(eltype(firstcol)), eltype(eltype(tail)))
    LeftArrowheadMatrix{T}(firstcol, tail)
end


struct LeftArrowheadLayout <: AbstractBandedBlockBandedLayout end
MemoryLayout(::Type{<:LeftArrowheadMatrix}) = LeftArrowheadLayout()

sublayout(::LeftArrowheadLayout, ::Type{<:NTuple{2,BlockSlice{<:BlockRange{1, Tuple{OneTo{Int}}}}}}) = LeftArrowheadLayout()
function sub_materialize(::LeftArrowheadLayout, V::AbstractMatrix)
    KR,JR = parentindices(V)
    P = parent(V)
    LeftArrowheadMatrix(P.firstcol, getindex.(P.tail, Ref(Int.(KR.block)), Ref(Base.OneTo(Int(JR.block[end])-1))))
end


function layout_replace_in_print_matrix(::LeftArrowheadLayout, A, i, j, s)
    bi = findblockindex.(axes(A), (i,j))
    I,J = block.(bi)
    i,j = blockindex.(bi)
    l,u = blockbandwidths(A)
    if J == Block(1) && Int(I) ≤ length(A.firstcol)
        return Base.replace_in_print_matrix(A.firstcol[Int(I)], i, j, s)
    elseif -l ≤ Int(J-I) ≤ u &&  i == j
        return s
    end
    Base.replace_with_centered_mark(s)
end
