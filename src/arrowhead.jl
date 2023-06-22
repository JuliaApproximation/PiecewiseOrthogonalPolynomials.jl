
"""
LeftArrowheadMatrix

    B   B   B  
    D   D   D  D
    …   …   …   …   …
    D   D   D  
"""
struct LeftArrowheadMatrix{T} <: AbstractBandedBlockBandedMatrix{T}
    toprow::Vector{BandedMatrix{T}}
    tail::Vector{BandedMatrix{T}}
end

function axes(L::LeftArrowheadMatrix)
    m = size(L.toprow[1],1)
    n = length(L.tail)
    μ,ν = size(L.tail[1])
    blockedrange(Vcat(m, Fill(n,μ))), blockedrange(Fill(n,ν))
end

function getindex(L::LeftArrowheadMatrix{T}, Kk::BlockIndex{1}, Jj::BlockIndex{1}) where T
    K,k = block(Kk),blockindex(Kk)
    J,j = block(Jj),blockindex(Jj)
    K == Block(1) && return L.toprow[Int(J)][k,j]
    k ≠ j && return zero(T)
    L.tail[k][Int(K)-1, Int(J)]
end

function getindex(L::LeftArrowheadMatrix, k::Int, j::Int)
    ax,bx = axes(L)
    L[findblockindex(ax, k), findblockindex(bx, j)]
end

function LeftArrowheadMatrix(toprow, tail)
    m = size(toprow[1],1)
    n = length(tail)
    l,u = bandwidths(tail[1])
    for op in toprow
        @assert size(op) == (m,n)
        @assert bandwidths(op) == (1,0)
    end
    for op in tail
        @assert bandwidths(op) == (l,u)
    end
    T = promote_type(eltype(eltype(toprow)), eltype(eltype(tail)))
    LeftArrowheadMatrix{T}(toprow, tail)
end


