
"""
ArrowheadMatrix

    A   B   B  
    C   D   D  D
    …   …   …   …   …
    C   D   D  
"""
struct ArrowheadMatrix{T, AA<:AbstractMatrix{T},
                      BB<:AbstractVector{<:AbstractMatrix{T}},
                      CC<:AbstractVector{<:AbstractMatrix{T}},
                      DD<:AbstractVector{<:AbstractMatrix{T}}} <: AbstractBandedBlockBandedMatrix{T}
    A::AA
    B::BB # first row blocks
    C::CC # first col blocks
    D::DD # these are interlaces

    ArrowheadMatrix{T, AA, BB, CC, DD}(A, B, C, D) where {T,AA,BB,CC,DD} = new{T,AA,BB,CC,DD}(A, B, C, D)
end

ArrowheadMatrix{T}(A, B, C, D) where T = ArrowheadMatrix{T, typeof(A), typeof(B), typeof(C), typeof(D)}(A, B, C, D)

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

copy(A::ArrowheadMatrix) = ArrowheadMatrix(copy(A.A), map(copy, A.B), map(copy, A.C), map(copy, A.D))

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

function MatrixFactorizations._reverse_chol!(A::ArrowheadMatrix, ::Type{UpperTriangular})
    @assert bandwidths(A.A) == (1,1)
    for B in A.B
        @assert bandwidths(B) == (1,0)
    end

    for B in A.D
        reversecholesky!(Symmetric(B))
    end
    
    ξ,n = size(A.A)
    m = length(A.D)
    @assert ξ == n == m+1
    # for each block interacting with B, and each entry of each
    # block
    for b = length(A.B):-1:1, k = m:-1:1
        for b̃ = b+1:length(A.B) # j loop
            Akj = A.D[k][b,b̃]'
            
            if !iszero(Akj) # often we have zeros so this avoids unnecessary computation
                @simd for i = k:k+1
                    A.B[b][i,k] -= A.B[b̃][i,k]*Akj
                end
            end
        end

        AkkInv = inv(copy(A.D[k][b,b]'))
        for i = k:k+1
            A.B[b][i,k] *= AkkInv'
        end
    end

    #(1,1) block update now
    for k = n:-1:1
        for b̃ = 1:length(A.B) # j loop
            if k ≠ 1
                j = k-1
                Akj = A.B[b̃][k,j]
                for i = j:j+1
                    A.A[i,k] -= A.B[b̃][i,j]*Akj
                end
            end

            if k ≠ n
                j = k
                Akj = A.B[b̃][k,j]
                i = j
                A.A[i,k] -= A.B[b̃][i,j]*Akj
            end
        end
    end

    reversecholesky!(Symmetric(A.A))

    # for b = 1:length(A.B)
    #     for j = 1:m
    #         AkkInv = inv(A.D[j][b,b])
    #         for k = j:j+1
    #             A.B[b][k,j] *= AkkInv'
    #         end
    #     end
    # end


    return UpperTriangular(A), convert(BlasInt, 0)
end