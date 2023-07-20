
"""
ArrowheadMatrix

    A   B   B
    C   D   D  D
    …   …   …   …   …
    C   D   D
"""
struct ArrowheadMatrix{T, AA<:AbstractMatrix{T}, BB, CC, DD} <: AbstractBandedBlockBandedMatrix{T}
    A::AA
    B::BB # first row blocks
    C::CC # first col blocks
    D::DD # these are interlaces

    ArrowheadMatrix{T, AA, BB, CC, DD}(A, B, C, D) where {T,AA,BB,CC,DD} = new{T,AA,BB,CC,DD}(A, B, C, D)
end

ArrowheadMatrix{T}(A, B, C, D) where T = ArrowheadMatrix{T, typeof(A), typeof(B), typeof(C), typeof(D)}(A, B, C, D)

const ArrowheadMatrices = Union{ArrowheadMatrix,Symmetric{<:Any,<:ArrowheadMatrix},Hermitian{<:Any,<:ArrowheadMatrix},
                                UpperOrLowerTriangular{<:Any,<:ArrowheadMatrix}}

                                

subblockbandwidths(A::ArrowheadMatrices) = (1,1)

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

for adj in (:adjoint, :transpose)
    @eval $adj(A::ArrowheadMatrix) = ArrowheadMatrix($adj(A.A), map($adj, A.C), map($adj, A.B), map($adj, A.D))
end

function getindex(L::ArrowheadMatrix{T}, Kk::BlockIndex{1}, Jj::BlockIndex{1})::T where T
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

    @assert 0 ≤ λ ≤ 1
    @assert 0 ≤ μ ≤ 1


    for op in B
        @assert size(op) == (ξ,m)
        λ,μ = bandwidths(op)
        @assert 0 ≤ λ ≤ 1
        @assert iszero(μ)
    end
    for op in C
        @assert size(op) == (m,n)
        λ,μ = bandwidths(op)
        @assert iszero(λ)
        @assert 0 ≤ μ ≤ 1
    end

    l,u = bandwidths(D[1])
    for op in D
        @assert bandwidths(op) == (l,u)
    end
    T = promote_type(eltype(A), mapreduce(eltype, promote_type, B; init=eltype(A)),
                     mapreduce(eltype, promote_type, C; init=eltype(A)), mapreduce(eltype, promote_type, D; init=eltype(A)))
    ArrowheadMatrix{T}(A, B, C, D)
end


struct ArrowheadLayout <: AbstractBandedBlockBandedLayout end
struct LazyArrowheadLayout <: AbstractLazyBandedBlockBandedLayout end
ArrowheadLayouts = Union{ArrowheadLayout,LazyArrowheadLayout,
                    SymmetricLayout{ArrowheadLayout},SymmetricLayout{LazyArrowheadLayout},
                    HermitianLayout{ArrowheadLayout},HermitianLayout{LazyArrowheadLayout},
                    TriangularLayout{'U', 'N', ArrowheadLayout}, TriangularLayout{'L', 'N', ArrowheadLayout},
                    TriangularLayout{'U', 'U', ArrowheadLayout}, TriangularLayout{'L', 'U', ArrowheadLayout},
                    TriangularLayout{'U', 'N', LazyArrowheadLayout}, TriangularLayout{'L', 'N', LazyArrowheadLayout},
                    TriangularLayout{'U', 'U', LazyArrowheadLayout}, TriangularLayout{'L', 'U', LazyArrowheadLayout}}
arrowheadlayout(_) = ArrowheadLayout()
arrowheadlayout(::BandedLazyLayouts) = LazyArrowheadLayout()
symmetriclayout(lay::ArrowheadLayouts) = SymmetricLayout{typeof(lay)}()

MemoryLayout(::Type{<:ArrowheadMatrix{<:Any,<:Any,<:Any,<:Any,<:AbstractVector{D}}}) where D = arrowheadlayout(MemoryLayout(D))

sublayout(::ArrowheadLayouts,
          ::Type{<:NTuple{2,BlockSlice{<:BlockRange{1, Tuple{OneTo{Int}}}}}}) = ArrowheadLayout()
function sub_materialize(::ArrowheadLayout, V::AbstractMatrix)
    KR,JR = parentindices(V)
    P = parent(V)
    M,N =  KR.block[end],JR.block[end]
    ArrowheadMatrix(P.A, P.B, P.C,
                    layout_getindex.(P.D, Ref(oneto(Int(M)-1)), Ref(oneto(Int(N)-1))))
end

symmetric(A) = Symmetric(A)
symmetric(A::Union{SymTridiagonal,Symmetric,Diagonal}) = A

function getproperty(F::Symmetric{<:Any,<:ArrowheadMatrix}, d::Symbol)
    P = getfield(F, :data)
    if d == :A
        return symmetric(P.A)
    elseif d == :B
        return P.B
    elseif d == :C
        adjoint.(P.B)
    elseif d == :D
        symmetric.(P.D)
    else
        getfield(F, d)
    end
end




function layout_replace_in_print_matrix(::ArrowheadLayouts, A, k, j, s)
    bi = findblockindex.(axes(A), (k,j))
    K,J = block.(bi)
    k,j = blockindex.(bi)
    K == J == Block(1) && return replace_in_print_matrix(A.A, k, j, s)
    if K == Block(1)
        return Int(J)-1 ≤ length(A.B) ? replace_in_print_matrix(A.B[Int(J)-1], k, j, s) : Base.replace_with_centered_mark(s)
    end
    if J == Block(1)
        return Int(K)-1 ≤ length(A.C) ? replace_in_print_matrix(A.C[Int(K)-1], k, j, s) : Base.replace_with_centered_mark(s)
    end
    k ≠ j && return Base.replace_with_centered_mark(s)
    return replace_in_print_matrix(A.D[k], Int(K)-1, Int(J)-1, s)
end

###
# Cholesky
####

function reversecholcopy(S::Symmetric{<:Any,<:ArrowheadMatrix})
    T = LinearAlgebra.choltype(S)
    A = parent(S)
    Symmetric(ArrowheadMatrix(LinearAlgebra.copymutable_oftype(A.A, T),
    LinearAlgebra.copymutable_oftype.(A.B, T), LinearAlgebra.copymutable_oftype.(A.C, T),
    LinearAlgebra.copymutable_oftype.(A.D, T)))
end

function MatrixFactorizations._reverse_chol!(A::ArrowheadMatrix, ::Type{UpperTriangular})

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

    return UpperTriangular(A), convert(BlasInt, 0)
end

tupleop(::Tuple{}, ::Tuple{}) = ()
tupleop(A::Tuple, B::Tuple{}) = A
tupleop(::typeof(+), A::Tuple{}, B::Tuple) = B
tupleop(::typeof(-), A::Tuple{}, B::Tuple) = -B
tupleop(op, A::Tuple, B::Tuple) = (op(first(A), first(B)), tupleop(op, tail(A), tail(B))...)


###
# Operations
###


for op in (:+, :-)
    @eval $op(A::ArrowheadMatrix, B::ArrowheadMatrix) = ArrowheadMatrix($op(A.A, B.A), tupleop($op, A.B, B.B), tupleop($op, A.C, B.C), $op(A.D, B.D))
end
-(A::ArrowheadMatrix) = ArrowheadMatrix(-A.A, map(-, A.B), map(-, A.C), -A.D)

###
# Triangular
###



for (UNIT, Tri) in (('U',UnitUpperTriangular), ('N', UpperTriangular))
    @eval @inline function materialize!(M::MatLdivVec{<:TriangularLayout{'U',$UNIT,ArrowheadLayout},
                                        <:AbstractStridedLayout})
        U,dest = M.A,M.B
        T = eltype(dest)

        P = triangulardata(U)



        ξ,n = size(P.A)
        A,B,D = P.A,P.B,P.D
        m = length(D)

        for k = 1:length(D)
            ArrayLayouts.ldiv!($Tri(D[k]), view(dest, n+k:m:length(dest)))
        end

        N = blocksize(P,1)

        # impose block structure
        b = PseudoBlockArray(dest, (axes(P,1),))
        b̃_1 = view(b, Block(1))

        for K = 1:min(N-1,length(B))
            muladd!(-one(T), B[K], view(b, Block(K+1)), one(T), b̃_1)
        end

        ArrayLayouts.ldiv!($Tri(A), b̃_1)

        dest
    end
end
for (UNIT, Tri) in (('U',UnitLowerTriangular), ('N', LowerTriangular))
    @eval @inline function materialize!(M::MatLdivVec{<:TriangularLayout{'L',$UNIT,ArrowheadLayout},
            <:AbstractStridedLayout})
        U,dest = M.A,M.B
        T = eltype(dest)

        P = triangulardata(U)
        ξ,n = size(P.A)
        A,C,D = P.A,P.C,P.D
        m = length(D)

        # impose block structure
        b = PseudoBlockArray(dest, (axes(P,1),))
        b̃_1 = view(b, Block(1))
        ArrayLayouts.ldiv!($Tri(A), b̃_1)

        N = blocksize(P,1)
        for K = 1:min(N-1,length(C))
            muladd!(-one(T), C[K], b̃_1, one(T), view(b, Block(K+1)))
        end


        for k = 1:length(D)
            ArrayLayouts.ldiv!($Tri(D[k]), view(dest, n+k:m:length(dest)))
        end

        dest
    end
end


for Tri in (:UpperTriangular, :UnitUpperTriangular)
    @eval function getproperty(F::$Tri{<:Any,<:ArrowheadMatrix}, d::Symbol)
        P = getfield(F, :data)
        if d == :A
            return $Tri(P.A)
        elseif d == :B
            return P.B
        elseif d == :C
            ()
        elseif d == :D
            $Tri.(P.D)
        else
            getfield(F, d)
        end
    end
end

for Tri in (:LowerTriangular, :UnitLowerTriangular)
    @eval function getproperty(F::$Tri{<:Any,<:ArrowheadMatrix}, d::Symbol)
        P = getfield(F, :data)
        if d == :A
            return $Tri(P.A)
        elseif d == :B
            return ()
        elseif d == :C
            P.C
        elseif d == :D
            $Tri.(P.D)
        else
            getfield(F, d)
        end
    end
end