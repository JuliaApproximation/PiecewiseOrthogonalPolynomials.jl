
"""
ArrowheadMatrix

    A   B   B  
    C   D   D  D
    …   …   …   …   …
    C   D   D  
"""
struct ArrowheadMatrix{T} <: AbstractBandedBlockBandedMatrix{T}
    A::BandedMatrix{T}
    B::Vector{BandedMatrix{T}} # first row blocks
    C::Vector{BandedMatrix{T}} # first col blocks
    D::Vector{BandedMatrix{T}} # these are interlaces
end

subblockbandwidths(A::ArrowheadMatrix) = bandwidths(A.A)
function blockbandwidths(A::ArrowheadMatrix)
    l,u = bandwidths(A.D[1])
    max(l,length(A.C)),max(u,length(A.B))
end

function axes(L::ArrowheadMatrix)
    ξ,n = size(L.A)
    m = length(L.D)
    μ,ν = size(L.D[1])
    blockedrange(Vcat(ξ, Fill(m,μ))), blockedrange(Vcat(n, Fill(m,ν)))
end

function getindex(L::ArrowheadMatrix{T}, Kk::BlockIndex{1}, Jj::BlockIndex{1}) where T
    K,k = block(Kk),blockindex(Kk)
    J,j = block(Jj),blockindex(Jj)
    J == K == Block(1) && return L.A[k,j]
    if K == Block(1)
        return Int(J)-1 ≤ length(L.B) ? L.B[Int(J)-1][k,j] : zero(T)
    end
    if J == Block(1)
        return Int(K)-1 ≤ length(L.C) ? L.C[Int(K)-1][k,j] : zero(T)
    end
    k ≠ j && return zero(T)
    return L.D[k][Int(K)-1, Int(J)-1]
end

function getindex(L::ArrowheadMatrix, k::Int, j::Int)
    ax,bx = axes(L)
    L[findblockindex(ax, k), findblockindex(bx, j)]
end

function ArrowheadMatrix(A, B, C, D)
    ξ,n = size(A)
    m = length(D)
    μ,ν = size(D[1])

    λ,μ = bandwidths(A)
    l,u = bandwidths(D[1])

    for op in B
        @assert size(op) == (ξ,m)
        λₖ,μₖ = bandwidths(op)
        @assert λₖ ≤ λ
        @assert μₖ ≤ μ
    end
    for op in C
        @assert size(op) == (m,n)
        λₖ,μₖ = bandwidths(op)
        @assert λₖ ≤ λ
        @assert μₖ ≤ μ
    end

    for op in D
        @assert bandwidths(op) == (l,u)
    end
    T = promote_type(eltype(A), eltype(eltype(B)), eltype(eltype(C)), eltype(eltype(D)))
    ArrowheadMatrix{T}(A, B, C, D)
end


struct ArrowheadLayout <: AbstractBandedBlockBandedLayout end
MemoryLayout(::Type{<:ArrowheadMatrix}) = ArrowheadLayout()

sublayout(::ArrowheadLayout, ::Type{<:NTuple{2,BlockSlice{<:BlockRange{1, Tuple{OneTo{Int}}}}}}) = ArrowheadLayout()
function sub_materialize(::ArrowheadLayout, V::AbstractMatrix)
    KR,JR = parentindices(V)
    P = parent(V)
    M,N =  KR.block[end],JR.block[end]
    ArrowheadMatrix(P.A, P.B[oneto(min(length(P.B),Int(N)-1))], P.C[oneto(min(length(P.B),Int(M)-1))],
                    getindex.(P.D, Ref(oneto(Int(M)-1)), Ref(oneto(Int(N)-1))))
end


function layout_replace_in_print_matrix(::ArrowheadLayout, A, k, j, s)
    bi = findblockindex.(axes(A), (k,j))
    K,J = block.(bi)
    k,j = blockindex.(bi)
    l,u = blockbandwidths(A)
    K == J == Block(1) && return replace_in_print_matrix(A.A, k, j, s)
    if K == Block(1)
        return Int(J)-1 ≤ length(A.B) ? replace_in_print_matrix(A.B[Int(J)-1], k, j, s) : Base.replace_with_centered_mark(s)
    end
    if J == Block(1)
        return Int(K)-1 ≤ length(A.C) ? replace_in_print_matrix(A.C[Int(K)-1], k, j, s) : Base.replace_with_centered_mark(s)
    end
    k ≠ j && return Base.replace_with_centered_mark(s)
    return replace_in_print_matrix(A.D[k], Int(K)-1, Int(J)-1, s)
end
