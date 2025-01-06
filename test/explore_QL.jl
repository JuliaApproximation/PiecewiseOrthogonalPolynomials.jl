using PiecewiseOrthogonalPolynomials, Plots, BlockArrays
using MatrixFactorizations, LinearAlgebra, BlockBandedMatrices
###
# QL
####
function my_ql(A::BBBArrowheadMatrix{T}) where T
    m,n = size(A.A)
    l = length(A.D)
    m2, n2 = size(A.D[1])
    @assert m == n == l+1
    @assert m2 == n2
    #results stored in F and tau
    F = BlockedArray(Matrix(A), axes(A))
    tau = zeros(m+l*m2)
    for j in m2:-1:3
        for i in l:-1:1
            upper_entry = F[Block(j-1, j+1)][i, i] #A.D[i][j-2,j]
            dia_entry = F[Block(j+1, j+1)][i, i] #A.D[i][j,j]
            #perform Householder transformation
            dia_entry_new = -sign(dia_entry)*sqrt(dia_entry^2 + upper_entry^2)
            v = [upper_entry, dia_entry-dia_entry_new]
            coef = 2/(v[1]^2+v[2]^2)
            #denote the householder transformation as [c1 s1;c2 s2]
            c1 = 1 - coef * v[1]^2
            s1 = - coef * v[1] * v[2]
            c2 = s1
            s2 = 1 - coef * v[2]^2
            print(dia_entry_new)
            F[m+(j-1)*l+i, m+(j-1)*l+i] = dia_entry_new #update F[Block(j+1, j+1)][i, i]
            F[m+(j-3)*l+i, m+(j-1)*l+i] = v[1]/v[2] #update F[Block(j-1, j+1)][i, i]
            tau[m+(j-1)*l+i] = coef*v[2]^2
            #row recombination(householder transformation) for other columns
            current_upper_entry = F[Block(j-1, j-1)][i, i] #A.D[i][j-2,j-2]
            current_lower_entry = F[Block(j+1, j-1)][i, i] #A.D[i][j,j-2]
            F[m+(j-3)*l+i, m+(j-3)*l+i] = c1 * current_upper_entry + s1 * current_lower_entry #update F[Block(j-1, j-1)][i, i]
            F[m+(j-1)*l+i, m+(j-3)*l+i] = c2 * current_upper_entry + s2 * current_lower_entry #update F[Block(j+1, j-1)][i, i]
            if j >= 5
                #Deal with A.D blocks which do not share common rows with A.C
                current_entry = F[Block(j-1, j-3)][i, i] #A.D[i][j-2,j-4]
                F[m+(j-3)*l+i, m+(j-5)*l+i] = c1 * current_entry #update F[Block(j-1, j-3)][i, i]
                F[m+(j-1)*l+i, m+(j-5)*l+i] = c2 * current_entry #update F[Block(j+1, j-3)][i, i]
            else
                #Deal with A.D blocks which share common rows with A.C
                current_entry = F[Block(j-1, 1)][i, i] #A.C[j-2][i,i]
                F[m+(j-3)*l+i, i] = c1 * current_entry #update F[Block(j-1, 1)][i, i]
                F[m+(j-1)*l+i, i] = c2 * current_entry #update F[Block(j+1, 1)][i, i]

                current_entry = F[Block(j-1, 1)][i, i+1] #A.C[j-2][i,i+1]
                F[m+(j-3)*l+i, i+1] = c1 * current_entry #update F[Block(j-1, 1)][i, i+1]
                F[m+(j-1)*l+i, i+1] = c2 * current_entry #F[Block(j+1, 1)][i, i+1]
            end
        end
    end

    #Deal with Block(1,3)
    #vectors x and Lambda denote a rank 1 semiseperable matrix
    lambda = 1.0
    Lambda = []
    x = [F[Block(1,3)][l+1,l]]
    x_len = abs(x[1])
    for i in l:-1:2 #consider i=1 later
        a = F[Block(1,3)][i,i]
        b = F[Block(1,3)][i,i-1]
        c = F[Block(3,3)][i,i]
        v_last = c + sign(c) * sqrt(a^2 + lambda^2 * x_len^2 + c^2)
        v_len = sqrt(a^2 + lambda^2 * x_len^2 + v_last^2)
        F[m+l+i,m+l+i] = -sign(c) * sqrt(a^2 + lambda^2 * x_len^2 + c^2)
        pushfirst!(Lambda, lambda / v_last)
        lambda = -2/v_len^2 * a * b * lambda
        F[m+l+i, m+l+i-1] = -2/v_len^2 * v_last * a * b
        x_first = (1 - 2/v_len^2 * a^2) * b / lambda
        pushfirst!(x, x_first)  
        x_len = sqrt(x_len^2 + x_first^2)  
        #record information of V
        F[i+1, m+l+i] = 0
        F[i, m+l+i] = a / v_last
        tau[m+l+i] = 2 * v_last^2 / v_len^2
    end
    #deal with the last column in Block(1,3)
    a = F[Block(1,3)][1,1]
    c = F[Block(3,3)][1,1]
    v_last = c + sign(c) * sqrt(a^2 + lambda^2 * x_len^2 + c^2)
    v_len = sqrt(a^2 + lambda^2 * x_len^2 + v_last^2)
    pushfirst!(Lambda, lambda / v_last)
    F[m+l+1,m+l+1] = -sign(c) * sqrt(a^2 + lambda^2 * x_len^2 + c^2)
    F[2, m+l+1] = 0
    F[1, m+l+1] = a / v_last
    tau[m+l+1] = 2 * v_last^2 / v_len^2

    F, tau, x, Lambda
end


𝐗 = range(-1,1; length=10)
C = ContinuousPolynomial{1}(𝐗)
plot(C[:,Block(2)])

#plot(C[:,Block.(2:3)])
M = C'C
#M = grammatrix(C)
Δ = weaklaplacian(C)
N = 6
KR = Block.(Base.OneTo(N))
Mₙ = M[KR,KR]
Δₙ = Δ[KR,KR]
A = Δₙ + 100^2 * Mₙ
FF,ttau, xx, LLambda = my_ql(A)
#tau = ql(A).τ
#f = ql(A).factors
