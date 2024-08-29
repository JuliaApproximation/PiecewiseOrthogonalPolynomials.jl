module BBBArrowheadMatrices
using LinearAlgebra, BlockArrays, BlockBandedMatrices, BandedMatrices, MatrixFactorizations, LazyBandedMatrices, LazyArrays, ArrayLayouts, InfiniteArrays, FillArrays
import ArrayLayouts: MemoryLayout, sublayout, sub_materialize, symmetriclayout, transposelayout, SymmetricLayout, HermitianLayout, TriangularLayout, layout_getindex, materialize!, MatLdivVec, AbstractStridedLayout, triangulardata, MatMulMatAdd, MatMulVecAdd, _fill_lmul!, layout_replace_in_print_matrix
import BandedMatrices: isbanded, bandwidths
import BlockArrays: BlockSlice, block, blockindex, blockvec
import BlockBandedMatrices: subblockbandwidths, blockbandwidths, AbstractBandedBlockBandedLayout, AbstractBandedBlockBandedMatrix
import Base: size, axes, getindex, +, -, *, /, ==, \, OneTo, oneto, replace_in_print_matrix, copy, diff, getproperty, adjoint, transpose, tail, _sum, inv, show, summary
import LazyArrays: paddeddata, AbstractLazyLayout
import LazyBandedMatrices: BlockBroadcastMatrix, BlockVec, AbstractLazyBandedBlockBandedLayout
import LinearAlgebra: BlasInt, eigvals, UpperOrLowerTriangular
import MatrixFactorizations: reversecholcopy
import FillArrays: AbstractFill
import FillArrays: SquareEye
import InfiniteArrays: OneToInf

import LazyBandedMatrices.LazyArraysBandedMatricesExt: BandedLazyLayouts


export BBBArrowheadMatrix

"""
BBBArrowheadMatrix

    A   B   B
    C   D   D  D
    …   …   …   …   …
    C   D   D
"""
struct BBBArrowheadMatrix{T, AA<:AbstractMatrix{T}, BB, CC, DD} <: AbstractBandedBlockBandedMatrix{T}
    A::AA
    B::BB # first row blocks
    C::CC # first col blocks
    D::DD # these are interlaces

    BBBArrowheadMatrix{T, AA, BB, CC, DD}(A, B, C, D) where {T,AA,BB,CC,DD} = new{T,AA,BB,CC,DD}(A, B, C, D)
end

BBBArrowheadMatrix{T}(A::AbstractMatrix{T}, B, C, D) where T = BBBArrowheadMatrix{T, typeof(A), typeof(B), typeof(C), typeof(D)}(A, B, C, D)
BBBArrowheadMatrix{T}(A, B, C, D) where T = BBBArrowheadMatrix(convert(AbstractMatrix{T}, A), B, C, D)

const ArrowheadMatrices = Union{BBBArrowheadMatrix,Symmetric{<:Any,<:BBBArrowheadMatrix},Hermitian{<:Any,<:BBBArrowheadMatrix},
                                UpperOrLowerTriangular{<:Any,<:BBBArrowheadMatrix}}

                                

subblockbandwidths(A::ArrowheadMatrices) = (1,1)

BlockArrays._show_typeof(io::IO, B::BBBArrowheadMatrix{T}) where T = print(io, "BBBArrowheadMatrix{$T}")

function blockbandwidths(A::BBBArrowheadMatrix)
    l,u = bandwidths(A.D[1])
    max(l,length(A.C)),max(u,length(A.B))
end

function axes(L::BBBArrowheadMatrix)
    ξ,n = size(L.A)
    m = length(L.D)
    μ,ν = size(L.D[1])
    blockedrange(Vcat(ξ, Fill(m,μ))), blockedrange(Vcat(n, Fill(m,ν)))
end

copy(A::BBBArrowheadMatrix) = BBBArrowheadMatrix(copy(A.A), map(copy, A.B), map(copy, A.C), map(copy, A.D))

for adj in (:adjoint, :transpose)
    @eval $adj(A::BBBArrowheadMatrix) = BBBArrowheadMatrix($adj(A.A), map($adj, A.C), map($adj, A.B), map($adj, A.D))
end

function getindex(L::BBBArrowheadMatrix{T}, Kk::BlockIndex{1}, Jj::BlockIndex{1})::T where T
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

function getindex(L::BBBArrowheadMatrix, k::Int, j::Int)
    ax,bx = axes(L)
    L[findblockindex(ax, k), findblockindex(bx, j)]
end

function BBBArrowheadMatrix(A, B, C, D)
    ξ,n = size(A)
    m = length(D)
    μ,ν = size(D[1])

    λ,μ = bandwidths(A)

    @assert -1 ≤ λ ≤ 1
    @assert -1 ≤ μ ≤ 1


    for op in B
        @assert size(op) == (ξ,m)
        λ,μ = bandwidths(op)
        @assert -1 ≤ λ ≤ 1
        @assert -1 ≤ μ ≤ 1
    end
    for op in C
        @assert size(op) == (m,n)
        λ,μ = bandwidths(op)
        @assert -1 ≤ λ ≤ 1
        @assert -1 ≤ μ ≤ 1
    end

    l,u = bandwidths(D[1])
    for op in D
        @assert bandwidths(op) == (l,u)
    end
    T = promote_type(eltype(A), mapreduce(eltype, promote_type, B; init=eltype(A)),
                     mapreduce(eltype, promote_type, C; init=eltype(A)), mapreduce(eltype, promote_type, D; init=eltype(A)))
    BBBArrowheadMatrix{T}(A, B, C, D)
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
arrowheadlayout(::DiagonalLayout{<:AbstractLazyLayout}) = LazyArrowheadLayout()
symmetriclayout(lay::ArrowheadLayouts) = SymmetricLayout{typeof(lay)}()

MemoryLayout(::Type{<:BBBArrowheadMatrix{<:Any,<:Any,<:Any,<:Any,<:AbstractVector{D}}}) where D = arrowheadlayout(MemoryLayout(D))
# Hack since ∞ diagonal fill is only DiagonalLayout{FillLayout}
MemoryLayout(::Type{<:BBBArrowheadMatrix{<:Any,<:Any,<:Any,<:Any,<:AbstractVector{<:Diagonal{<:Any,<:AbstractFill{<:Any,1,<:Tuple{OneToInf}}}}}}) = LazyArrowheadLayout()

sublayout(::ArrowheadLayouts,
          ::Type{<:NTuple{2,BlockSlice{<:BlockRange{1, Tuple{OneTo{Int}}}}}}) = ArrowheadLayout()
function sub_materialize(::ArrowheadLayout, V::AbstractMatrix)
    KR,JR = parentindices(V)
    P = parent(V)
    M,N =  KR.block[end],JR.block[end]
    BBBArrowheadMatrix(P.A, P.B, P.C,
                    layout_getindex.(P.D, Ref(oneto(Int(M)-1)), Ref(oneto(Int(N)-1))))
end

symmetric(A) = Symmetric(A)
symmetric(A::Union{SymTridiagonal,Symmetric,Diagonal}) = A

function getproperty(F::Symmetric{<:Any,<:BBBArrowheadMatrix}, d::Symbol)
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

####
# Mul
####

function materialize!(M::MatMulVecAdd{<:ArrowheadLayouts,<:AbstractStridedLayout,<:AbstractStridedLayout})
    α, A, x_in, β, y_in = M.α, M.A, M.B, M.β, M.C
    x = BlockedArray(x_in, (axes(A,2), ))
    y = BlockedArray(y_in, (axes(A,1),))
    m,n = size(A.A)

    _fill_lmul!(β, y)

    mul!(view(y, Block(1)), A.A, view(x, Block(1)), α, one(α))
    for k = 1:length(A.B)
        mul!(view(y, Block(1)), A.B[k], view(x, Block(k+1)), α, one(α))
    end
    for k = 1:length(A.C)
        mul!(view(y, Block(k+1)), A.C[k], view(x, Block(1)), α, one(α))
    end

    d = length(A.D)
    for k = 1:d
        mul!(view(y, m+k:d:size(y,1)), A.D[k], view(x, n+k:d:size(x,1)), α, one(α))
    end
    return y_in
end

function materialize!(M::MatMulMatAdd{<:ArrowheadLayouts,<:AbstractColumnMajor,<:AbstractColumnMajor})
    α, A, X_in, β, Y_in = M.α, M.A, M.B, M.β, M.C
    X = BlockedArray(X_in, (axes(A,2), axes(X_in,2)))
    Y = BlockedArray(Y_in, (axes(A,1), axes(X_in,2)))
    m,n = size(A.A)

    _fill_lmul!(β, Y)
    for J = blockaxes(X,2)
        mul!(view(Y, Block(1), J), A.A, view(X, Block(1), J), α, one(α))
        for k = 1:min(length(A.B), blocksize(X,1)-1)
            mul!(view(Y, Block(1), J), A.B[k], view(X, Block(k+1), J), α, one(α))
        end
        for k = 1:min(length(A.C), blocksize(Y,1)-1)
            mul!(view(Y, Block(k+1), J), A.C[k], view(X, Block(1), J), α, one(α))
        end
    end
    d = length(A.D)
    for k = 1:d
        mul!(view(Y, m+k:d:size(Y,1), :), A.D[k], view(X, n+k:d:size(Y,1), :), α, one(α))
    end
    return Y_in
end

function materialize!(M::MatMulMatAdd{<:AbstractColumnMajor,<:ArrowheadLayouts,<:AbstractColumnMajor})
    α, X_in, A, β, Y_in = M.α, M.A, M.B, M.β, M.C
    X = BlockedArray(X_in, (axes(X_in,1), axes(A,1)))
    Y = BlockedArray(Y_in, (axes(Y_in,1), axes(A,2)))
    m,n = size(A.A)

    _fill_lmul!(β, Y)
    for K = blockaxes(X,1)
        mul!(view(Y, K, Block(1)), view(X, K, Block(1)), A.A, α, one(α))
        for k = 1:length(A.C)
            mul!(view(Y, K, Block(1)), view(X, K, Block(k+1)), A.C[k], α, one(α))
        end
        for k = 1:length(A.B)
            mul!(view(Y, K, Block(k+1)), view(X, K, Block(1)), A.B[k], α, one(α))
        end
    end
    d = length(A.D)
    for k = 1:d
        mul!(view(Y, :, n+k:d:size(Y,2)), view(X, :, m+k:d:size(Y,2)), A.D[k], α, one(α))
    end
    return Y_in
end



###
# Cholesky
####

function reversecholcopy(S::Symmetric{<:Any,<:BBBArrowheadMatrix})
    T = LinearAlgebra.choltype(S)
    A = parent(S)
    Symmetric(BBBArrowheadMatrix(LinearAlgebra.copymutable_oftype(A.A, T),
    LinearAlgebra.copymutable_oftype.(A.B, T), LinearAlgebra.copymutable_oftype.(A.C, T),
    LinearAlgebra.copymutable_oftype.(A.D, T)))
end



function MatrixFactorizations.reversecholesky_layout!(::ArrowheadLayouts, A::AbstractMatrix, ::Type{UpperTriangular})
    Threads.@threads for B in A.D
        reversecholesky!(Symmetric(B))
    end

    if !isempty(A.B)
        if bandwidths(A.B[1]) == (1,0)
            _reverse_chol_lower_B!(A, UpperTriangular)
        else
            _reverse_chol_upper_B!(A, UpperTriangular)
        end
    end

    reversecholesky!(Symmetric(A.A))

    return UpperTriangular(A), convert(BlasInt, 0)
end


# This is the case that the off-diagonal Bs are lower triangular
# which is the case with Neumann conditions
function _reverse_chol_lower_B!(A, ::Type{UpperTriangular})
    for B in A.B
        @assert bandwidths(B) == (1,0)
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
end

function _reverse_chol_upper_B!(A, ::Type{UpperTriangular})
    for B in A.B
        @assert bandwidths(B) == (0,1)
    end

    ξ,n = size(A.A)
    m = length(A.D)

    N = blocksize(A,1)

    @assert ξ == n == m-1
    # for each block interacting with B, and each entry of each
    # block
    for b = min(length(A.B),N-1):-1:1, k = m:-1:1
        for b̃ = b+1:min(length(A.B),N-1) # j loop
            Akj = A.D[k][b,b̃]'

            if !iszero(Akj) # often we have zeros so this avoids unnecessary computation
                @simd for i = max(1,k-1):min(k,n)
                    A.B[b][i,k] -= A.B[b̃][i,k]*Akj
                end
            end
        end

        AkkInv = inv(copy(A.D[k][b,b]'))
        for i = max(1,k-1):min(k,n)
            A.B[b][i,k] *= AkkInv'
        end
    end

    #(1,1) block update now
    # k == n, only contribution is from column n+1
    for b̃ = 1:length(A.B) # j loop
        for k = n:-1:1
            Akj = A.B[b̃][k,k+1]
            A.A[k,k] -= Akj^2

            Akj = A.B[b̃][k,k]
            for i = max(1,k-1):k
                A.A[i,k] -= Akj * A.B[b̃][i,k]
            end
        end
    end
end


tupleop(::typeof(+), ::Tuple{}, ::Tuple{}) = ()
tupleop(::typeof(-), ::Tuple{}, ::Tuple{}) = ()
tupleop(::typeof(+), A::Tuple, B::Tuple{}) = A
tupleop(::typeof(-), A::Tuple, B::Tuple{}) = A
tupleop(::typeof(+), A::Tuple{}, B::Tuple) = B
tupleop(::typeof(-), A::Tuple{}, B::Tuple) = map(-,B)
tupleop(op, A::Tuple, B::Tuple) = (op(first(A), first(B)), tupleop(op, tail(A), tail(B))...)


###
# Operations
###


for op in (:+, :-)
    @eval $op(A::BBBArrowheadMatrix, B::BBBArrowheadMatrix) = BBBArrowheadMatrix($op(A.A, B.A), tupleop($op, A.B, B.B), tupleop($op, A.C, B.C), $op(A.D, B.D))
end
-(A::BBBArrowheadMatrix) = BBBArrowheadMatrix(-A.A, map(-, A.B), map(-, A.C), -A.D)

for op in (:*, :\)
    @eval $op(c::Number, A::BBBArrowheadMatrix) = BBBArrowheadMatrix($op(c, A.A), broadcast($op, c, A.B), broadcast($op, c, A.C), broadcast($op, c, A.D))
end

for op in (:*, :/)
    @eval $op(A::BBBArrowheadMatrix, c::Number) = BBBArrowheadMatrix($op(A.A, c), broadcast($op, A.B, c), broadcast($op, A.C, c), broadcast($op, A.D, c))
end

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

        Threads.@threads for k = 1:m
            ArrayLayouts.ldiv!($Tri(D[k]), view(dest, n+k:m:length(dest)))
        end

        N = blocksize(P,1)

        # impose block structure
        b = BlockedArray(dest, (axes(P,1),))
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
        b = BlockedArray(dest, (axes(P,1),))
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
    @eval function getproperty(F::$Tri{<:Any,<:BBBArrowheadMatrix}, d::Symbol)
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
    @eval function getproperty(F::$Tri{<:Any,<:BBBArrowheadMatrix}, d::Symbol)
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

####
# banded interface
#####
bbbbandwidth(k, A) = bandwidth(A, k)
bbbbandwidth(k, A, C...) = +(size(A,k), size.(Base.front(C), k)...) + bandwidth(last(C),k)
bandwidths(A::BBBArrowheadMatrix) = bbbbandwidth(1, A.A, A.C...),  bbbbandwidth(2, A.A, A.B...)
isbanded(A::BBBArrowheadMatrix) = true


####
# eigvals
#####

eigvals(A::Symmetric{<:Real,<:BBBArrowheadMatrix}) = eigvals(Symmetric(BandedMatrix(parent(A))))
eigvals(A::Symmetric{<:Real,<:BBBArrowheadMatrix}, B::Symmetric{<:Real,<:BBBArrowheadMatrix}) = eigvals(Symmetric(BandedMatrix(parent(A))), Symmetric(BandedMatrix(parent(B))))

end #module