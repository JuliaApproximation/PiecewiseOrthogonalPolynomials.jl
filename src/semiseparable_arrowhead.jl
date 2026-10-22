struct SemiseparableBBBArrowheadMatrix{T} <: AbstractBlockBandedMatrix{T}
    # banded parts
    A::BandedMatrix{T}
    B::NTuple{2,BandedMatrix{T}} # first row blocks
    C::NTuple{4,BandedMatrix{T}} # first col blocks
    D

    # fill parts
    Asub::NTuple{2,Vector{T}}
    Asup::Tuple{2,Matrix{T}} # matrices are m × 2

    Bsub::NTuple{2,Vector{T}}
    Bsup::NTuple{2,NTuple{2,Vector{T}}}

    Csub::NTuple{2,NTuple{2,Vector{T}}}
    Csup::NTuple{2,Vector{T}}

    A22sub::NTuple{2,Vector{T}}
    A32sub::NTuple{2,Vector{T}}

    A32extra::Vector{T}
    A33extra::Vector{T}

    D::DD # these are interlaces

end

axes(::SemiseparableBBBArrowheadMatrix) = ...

function getindex(L::SemiseparableBBBArrowheadMatrix{T}, Kk::BlockIndex{1}, Jj::BlockIndex{1})::T where T
    K,k = block(Kk),blockindex(Kk)
    J,j = block(Jj),blockindex(Jj)
    # TODO: add getindex
end


function getindex(L::SemiseparableBBBArrowheadMatrix, k::Int, j::Int)
    ax,bx = axes(L)
    L[findblockindex(ax, k), findblockindex(bx, j)]
end

