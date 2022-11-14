using SparseArrays
using PyPlot
using LinearAlgebra
using Random
using Optim
using MAT
using Interpolations
using NLopt
using LBFGSB
using Statistics

cd( @__DIR__)

function meshgrid(x, y)
    X = [x for _ in y, x in x]
    Y = [y for y in y, _ in x]
    X, Y
 end

@inline  function etaisyys(x,y)
    return sqrt(x^2 + y^2)
end


function regmatrices_first(dim)
    reg1d = spdiagm(Pair(0,-1*ones(dim))) + spdiagm(Pair(1,ones(dim-1))) + spdiagm(Pair(-dim+1,ones(1))) ;reg1d[dim,dim] = 0
    #reg1d = reg1d[1:dim-1,:]
    iden = I(dim)
    regx = kron(reg1d,iden)
    regy = kron(iden,reg1d)

    rmxix = sum(abs.(regx) ,dims=2) .< 2
    rmyix = sum(abs.(regy) ,dims=2) .< 2
    boundary = ((rmxix + rmyix)[:]) .!= 0
    q = findall(boundary .== 1)
    regx = regx[setdiff(1:dim^2,q), :] 
    regy = regy[setdiff(1:dim^2,q), :] 
    
    s = length(q)
    bmatrix = sparse(zeros(s,dim*dim))
    for i=1:s
        v = q[i]
        bmatrix[i,v] = 1
    end
    #bmatrix = bmatrix[i,i] .= 1

    return regx,regy,bmatrix
end

function spdematrix(xs,a,b)
    dx = abs(xs[2]-xs[1])
    N = length(xs)
    
    M = sparse(zeros(N,N))
    for i=2:N-1
        M[i,i-1] = -a/dx^2;
        M[i,i+1] = -a/dx^2;
        M[i,i] = 2*a/dx^2;# +b;

    end

    M[1,1] = 1 # Option 1
    M[N,N] = 1
    M[N,N-1] = -1 

    #M[1,:] = [2*a/dx^2;-a/dx^2; zeros(N-3);-a/dx^2]; # Option 2
    #M[N,:] = [-a/dx^2; zeros(N-3); -a/dx^2;2*a/dx^2];
    
    M = kron(M,I(N)) + kron(I(N),M)
    M = M + I(size(M,2))*b;

    return M
end    

#=
function solution3(coef,F)

    Na = Int64(sqrt(length(coef)))
    N = Na - 2
    M = zeros(N^2,N^2)
    ixs = LinearIndices((1:N,1:N))
    ixsNa = LinearIndices((1:Na,1:Na))
    h = 1/(N-1)
    y = zeros(N*N)

    @inline function ix1(ix1s,ix2s)
        return ixs[ix1s,ix2s]
    end

    @inline function ix2(ix1s,ix2s)
        return ixsNa[ix1s+1,ix2s+1]
    end

    # rows = Vector{Int64}(undef,0)
    # cols = Vector{Int64}(undef,0)
    # vals = Vector{Float64}(undef,0)

    for i = 2:N-1
       
        for j = 2:N-1

            y[ix1(i,j)] = F[ix2(i,j)]
            
            # append!(rows,ixs[i,j]); append!(cols,ixs[i,j]); append!(vals,4*coef[ixs[i,j]]/h^2);
            # append!(rows,ixs[i,j]); append!(cols,ixs[i-1,j]); append!(vals,-coef[ixs[i,j]]/h^2);
            # append!(rows,ixs[i,j]); append!(cols,ixs[i,j-1]); append!(vals,-coef[ixs[i,j]]/h^2);
            # append!(rows,ixs[i,j]); append!(cols,ixs[i+1,j]); append!(vals,-coef[ixs[i,j]]/h^2);
            # append!(rows,ixs[i,j]); append!(cols,ixs[i,j+1]); append!(vals,-coef[ixs[i,j]]/h^2);

            # append!(rows,ixs[i,j]); append!(cols,ixs[i+1,j]); append!(vals,-(coef[ixs[i+1,j]]/(4h*h) - coef[ixs[i-1,j]]/(4h*h)));
            # append!(rows,ixs[i,j]); append!(cols,ixs[i-1,j]); append!(vals,+(coef[ixs[i+1,j]]/(4h*h) -coef[ixs[i-1,j]]/(4h*h)));
            # append!(rows,ixs[i,j]); append!(cols,ixs[i,j+1]); append!(vals,-(coef[ixs[i,j+1]]/(4h*h) - coef[ixs[i,j-1]]/(4h*h)));
            # append!(rows,ixs[i,j]); append!(cols,ixs[i,j-1]); append!(vals,+(coef[ixs[i,j+1]]/(4h*h) -coef[ixs[i,j-1]]/(4h*h)));

            M[ix1(i,j),ix1(i,j)] += 4*coef[ix2(i,j)]/h^2
            M[ix1(i,j),ix1(i-1,j)] += -coef[ix2(i,j)]/h^2
            M[ix1(i,j),ix1(i,j-1)] += -coef[ix2(i,j)]/h^2
            M[ix1(i,j),ix1(i+1,j)] += -coef[ix2(i,j)]/h^2
            M[ix1(i,j),ix1(i,j+1)] += -coef[ix2(i,j)]/h^2


            M[ix1(i,j),ix1(i+1,j)] += -(coef[ix2(i+1,j)]/(4h*h) - coef[ix2(i-1,j)]/(4h*h))
            M[ix1(i,j),ix1(i-1,j)] += +(coef[ix2(i+1,j)]/(4h*h) -coef[ix2(i-1,j)]/(4h*h))
            M[ix1(i,j),ix1(i,j+1)] += -(coef[ix2(i,j+1)]/(4h*h) - coef[ix2(i,j-1)]/(4h*h))
            M[ix1(i,j),ix1(i,j-1)] += +(coef[ix2(i,j+1)]/(4h*h) -coef[ix2(i,j-1)]/(4h*h))


          
  
        end
    end

    #M = sparse(rows,cols,vals,N^2,N^2)
    for i = 2:N-1

        # append!(rows,ixs[i,1]); append!(cols,ixs[i,1]); append!(vals,1.0);
        # append!(rows,ixs[i,N]); append!(cols,ixs[i,N]); append!(vals,1.0);
        # append!(rows,ixs[1,i]); append!(cols,ixs[1,i]); append!(vals,1.0);
        # append!(rows,ixs[N,i]); append!(cols,ixs[N,i]); append!(vals,1.0);


        # M[ixs[i,1],ixs[i,1]] = 1
        # M[ixs[i,N],ixs[i,N]] = 1

        # M[ixs[1,i],ixs[1,i]] = 1
        # M[ixs[N,i],ixs[N,i]] = 1

        # y[ixs[i,1]] = 0
        # y[ixs[i,N]] = 0
        # y[ixs[1,i]] = 0
        # y[ixs[N,i]] = 0

        M[ix1(1,i),ix1(1,i)] += 4*coef[ix2(1,i)]/h^2
        M[ix1(1,i),ix1(1,i-1)] += -coef[ix2(1,i)]/h^2
        M[ix1(1,i),ix1(1,i+1)] += -coef[ix2(1,i)]/h^2
        M[ix1(1,i),ix1(2,i)] += -coef[ix2(1,i)]/h^2

        M[ix1(1,i),ix1(2,i)] += -(coef[ix2(1+1,i)]/(4h*h) - coef[ix2(1-1,i)]/(4h*h))
        M[ix1(1,i),ix1(1,i+1)] += -(coef[ix2(1,i+1)]/(4h*h) - coef[ix2(1,i-1)]/(4h*h))
        M[ix1(1,i),ix1(1,i-1)] += +(coef[ix2(1,i+1)]/(4h*h) -coef[ix2(1,i-1)]/(4h*h))

        y[ix1(1,i)] = F[ix2(1,i)] + coef[ix2(1,i)]/h^2 -(coef[ix2(1+1,i)]/(4h*h) -coef[ix2(1-1,i)]/(4h*h))


        M[ix1(N,i),ix1(N,i)] += 4*coef[ix2(N,i)]/h^2
        M[ix1(N,i),ix1(N,i-1)] += -coef[ix2(N,i)]/h^2
        M[ix1(N,i),ix1(N,i+1)] += -coef[ix2(N,i)]/h^2
        M[ix1(N,i),ix1(N-1,i)] += -coef[ix2(N,i)]/h^2

        M[ix1(N,i),ix1(N-1,i)] += +(coef[ix2(N+1,i)]/(4h*h) - coef[ix2(N-1,i)]/(4h*h))
        M[ix1(N,i),ix1(N,i+1)] += -(coef[ix2(N,i+1)]/(4h*h) - coef[ix2(N,i-1)]/(4h*h))
        M[ix1(N,i),ix1(N,i-1)] += +(coef[ix2(N,i+1)]/(4h*h) -coef[ix2(N,i-1)]/(4h*h))

        y[ix1(N,i)] = F[ix2(N,i)] + coef[ix2(N,i)]/h^2 -(coef[ix2(N+1,i)]/(4h*h) -coef[ix2(N-1,i)]/(4h*h))


        M[ix1(i,1),ix1(i,1)] += 4*coef[ix2(i,1)]/h^2
        M[ix1(i,1),ix1(i,i+1)] += -coef[ix2(i,1)]/h^2
        M[ix1(i,1),ix1(i-1,i)] += -coef[ix2(i,1)]/h^2
        M[ix1(i,1),ix1(i+1,i)] += -coef[ix2(i,1)]/h^2

        M[ix1(N,i),ix1(N-1,i)] += +(coef[ix2(N+1,i)]/(4h*h) - coef[ix2(N-1,i)]/(4h*h))
        M[ix1(N,i),ix1(N,i+1)] += -(coef[ix2(N,i+1)]/(4h*h) - coef[ix2(N,i-1)]/(4h*h))
        M[ix1(N,i),ix1(N,i-1)] += +(coef[ix2(N,i+1)]/(4h*h) -coef[ix2(N,i-1)]/(4h*h))

        y[ix1(N,i)] = F[ix2(N,i)] + coef[ix2(i,1)]/h^2 -(coef[ix2(N+1,i)]/(4h*h) -coef[ix2(N-1,i)]/(4h*h))

    end
    
    
    return M,y
end
=#
function solution(X,Y,coef,F)

    N = Int64(sqrt(length(coef)))
    #M = zeros(N^2,N^2)
    ixs = LinearIndices((1:N,1:N))
    h = X[1,2]-X[1,1]
    y = zeros(N*N)

    rows = Vector{Int64}(undef,0)
    cols = Vector{Int64}(undef,0)
    vals = Vector{Float64}(undef,0)

    for i = 2:N-1
       
        for j = 2:N-1

            y[ixs[i,j]] = F[ixs[i,j]]
            
            append!(rows,ixs[i,j]); append!(cols,ixs[i,j]); append!(vals,4*coef[ixs[i,j]]/h^2);
            append!(rows,ixs[i,j]); append!(cols,ixs[i-1,j]); append!(vals,-coef[ixs[i,j]]/h^2);
            append!(rows,ixs[i,j]); append!(cols,ixs[i,j-1]); append!(vals,-coef[ixs[i,j]]/h^2);
            append!(rows,ixs[i,j]); append!(cols,ixs[i+1,j]); append!(vals,-coef[ixs[i,j]]/h^2);
            append!(rows,ixs[i,j]); append!(cols,ixs[i,j+1]); append!(vals,-coef[ixs[i,j]]/h^2);

            append!(rows,ixs[i,j]); append!(cols,ixs[i+1,j]); append!(vals,-(coef[ixs[i+1,j]]/(4h*h) - coef[ixs[i-1,j]]/(4h*h)));
            append!(rows,ixs[i,j]); append!(cols,ixs[i-1,j]); append!(vals,+(coef[ixs[i+1,j]]/(4h*h) -coef[ixs[i-1,j]]/(4h*h)));
            append!(rows,ixs[i,j]); append!(cols,ixs[i,j+1]); append!(vals,-(coef[ixs[i,j+1]]/(4h*h) - coef[ixs[i,j-1]]/(4h*h)));
            append!(rows,ixs[i,j]); append!(cols,ixs[i,j-1]); append!(vals,+(coef[ixs[i,j+1]]/(4h*h) -coef[ixs[i,j-1]]/(4h*h)));

            # M[ixs[i,j],ixs[i,j]] += 4*coef[ixs[i,j]]/h^2
            # M[ixs[i,j],ixs[i-1,j]] += -coef[ixs[i,j]]/h^2
            # M[ixs[i,j],ixs[i,j-1]] += -coef[ixs[i,j]]/h^2
            # M[ixs[i,j],ixs[i+1,j]] += -coef[ixs[i,j]]/h^2
            # M[ixs[i,j],ixs[i,j+1]] += -coef[ixs[i,j]]/h^2


            # M[ixs[i,j],ixs[i+1,j]] += -(coef[ixs[i+1,j]]/(4h*h) - coef[ixs[i-1,j]]/(4h*h))
            # M[ixs[i,j],ixs[i-1,j]] += +(coef[ixs[i+1,j]]/(4h*h) -coef[ixs[i-1,j]]/(4h*h))
            # M[ixs[i,j],ixs[i,j+1]] += -(coef[ixs[i,j+1]]/(4h*h) - coef[ixs[i,j-1]]/(4h*h))
            # M[ixs[i,j],ixs[i,j-1]] += +(coef[ixs[i,j+1]]/(4h*h) -coef[ixs[i,j-1]]/(4h*h))


            # M[ixs[i,j],ixs[i,j]] += ((coef[ixs[i-1,j]]+coef[ixs[i,j]])/2 + (coef[ixs[i,j]]+coef[ixs[i+1,j]])/2 + (coef[ixs[i,j]]+coef[ixs[i,j+1]])/2 + (coef[ixs[i,j-1]]+coef[ixs[i,j]])/2)/h^2
            # M[ixs[i,j],ixs[i+1,j]] += -(coef[ixs[i,j]]/(2h*h) + coef[ixs[i+1,j]]/(2h*h))
            # M[ixs[i,j],ixs[i-1,j]] += -(coef[ixs[i,j]]/(2h*h) + coef[ixs[i+1,j]]/(2h*h))
            # M[ixs[i,j],ixs[i,j+1]] += -(coef[ixs[i,j]]/(2h*h) + coef[ixs[i,j+1]]/(2h*h))
            # M[ixs[i,j],ixs[i,j-1]] += -(coef[ixs[i,j]]/(2h*h) + coef[ixs[i,j+1]]/(2h*h))



            # M[ixs[i,j],ixs[i+1,j]] += -(coef[ixs[i+1,j]]/(h*h) - coef[ixs[i,j]]/(h*h))
            # M[ixs[i,j],ixs[i,j]] += coef[ixs[i+1,j]]/(h*h) -coef[ixs[i,j]]/(h*h)
            # M[ixs[i,j],ixs[i,j+1]] += -(coef[ixs[i,j+1]]/(h*h) - coef[ixs[i,j]]/(h*h))
            # M[ixs[i,j],ixs[i,j]] += coef[ixs[i,j+1]]/(h*h) -coef[ixs[i,j]]/(h*h)


            # M[ixs[i,j],ixs[i,j]] += -(coef[ixs[i,j]]/(h*h) - coef[ixs[i-1,j]]/(h*h))
            # M[ixs[i,j],ixs[i-1,j]] += coef[ixs[i,j]]/(h*h) -coef[ixs[i-1,j]]/(h*h)
            # M[ixs[i,j],ixs[i,j]] += -(coef[ixs[i,j]]/(h*h) - coef[ixs[i,j-1]]/(h*h))
            # M[ixs[i,j],ixs[i,j-1]] += coef[ixs[i,j]]/(h*h) -coef[ixs[i,j-1]]/(h*h)
  
        end
    end

   
    #B = Diagonal(zeros(N^2))
    #Boff = I(N^2)
    # for i = 1:N
  
    #     # M[ixs[i,1],ixs[i,1]] = 1
    #     # Boff[ixs[i,1],ixs[i,1]] = 0
    #     # B[ixs[i,1],ixs[i,1]] = 1.0
    #     y[ixs[i,1]] = 0
    #     append!(rows,ixs[i,1]); append!(cols,ixs[i,1]); append!(vals,1.0);

    # end

    # for i = 1:N
  
    #     # M[ixs[i,N],ixs[i,N]] = 1
    #     # Boff[ixs[i,N],ixs[i,N]] = 0
    #     # B[ixs[i,N],ixs[i,N]] = 1.0
    #     y[ixs[i,N]] = 0
    #     append!(rows,ixs[i,N]); append!(cols,ixs[i,N]); append!(vals,1.0);
     
    # end

    # for i = 2:N-1

    #     # k*x_{2,i} - 2*h*g_{1,i}  = k*x{0,i}
    #     # x{0,i}*-coef[ixs[1,i]]/h^2 = (x_{2,i} - 2*h*g_{1,i})*-coef[ixs[1,i]]/h^2 

    #     gv = 1

    #     # append!(rows,ixs[1,i]); append!(cols,ixs[1,i]); append!(vals,-1);
    #     # append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,1);
    #     # y[ixs[1,i]] = 0

    #     append!(rows,ixs[1,i]); append!(cols,ixs[1,i]); append!(vals,4*coef[ixs[1,i]]/h^2);
    #     append!(rows,ixs[1,i]); append!(cols,ixs[1,i+1]); append!(vals,-coef[ixs[1,i]]/h^2);
    #     append!(rows,ixs[1,i]); append!(cols,ixs[1,i-1]); append!(vals,-coef[ixs[1,i]]/h^2);
    #     append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,-coef[ixs[1,i]]/h^2);
    #     append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,-coef[ixs[1,i]]/h^2);
    #     ##append!(rows,ixs[1,i]); append!(cols,ixs[1-1,i]); append!(vals,-coef[ixs[1,i]]/h^2);

    #     ##append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,-(coef[ixs[1+1,i]]/(4h*h) - coef[ixs[1-1,i]]/(4h*h)));
    #     ##append!(rows,ixs[1,i]); append!(cols,ixs[1-1,i]); append!(vals,+(coef[ixs[1+1,j]]/(4h*h) -coef[ixs[1-1,i]]/(4h*h)));
    #     #append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,-(coef[ixs[1+1,i]]/(2h*h) - coef[ixs[1,i]]/(2h*h)));
    #     #append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,+(coef[ixs[1+1,i]]/(2h*h) - coef[ixs[1,i]]/(2h*h)));
    #     append!(rows,ixs[1,i]); append!(cols,ixs[1,i+1]); append!(vals,-(coef[ixs[1,i+1]]/(4h*h) - coef[ixs[1,i-1]]/(4h*h)));
    #     append!(rows,ixs[1,i]); append!(cols,ixs[1,i-1]); append!(vals,+(coef[ixs[1,i+1]]/(4h*h) -coef[ixs[1,i-1]]/(4h*h)));


    #     y[ixs[1,i]] =  F[ixs[1,i]] + 2*gv/h

    #     # M[ixs[1,i],ixs[1,i]] = 1
    #     #Boff[ixs[1,i],ixs[1,i]] = 0
    #     #B[ixs[1,i],ixs[1,i]] = 1.0
    #     #y[ixs[1,i]] = 0

    # end
    
    # for i = 1:N
  
    #     # M[ixs[N,i],ixs[N,i]] = 1
    #     #Boff[ixs[N,i],ixs[N,i]] = 0
    #     #B[ixs[N,i],ixs[N,i]] = 1.0
    #     y[ixs[N,i]] = 0

    #     append!(rows,ixs[N,i]); append!(cols,ixs[N,i]); append!(vals,1.0);

    # end
    
    boundary = Set([ixs[1,i] for i in 1:N])
    push!(boundary, [ixs[N,i] for i in 1:N]...)
    push!(boundary, [ixs[i,1] for i in 1:N]...)
    push!(boundary, [ixs[i,N] for i in 1:N]...)

    for b in boundary
        append!(rows,ixs[b]); append!(cols,ixs[b]); append!(vals,1.0);
        y[ixs[b]] = 0
    end
    
    M = sparse(rows,cols,vals,N^2,N^2)
    return M,y
    #return Boff*M + B,y
end

function solutionexp(X,Y,coef,F)

    N = Int64(sqrt(length(coef)))
    #M = zeros(N^2,N^2)
    ixs = LinearIndices((1:N,1:N))
    h = X[1,2]-X[1,1]
    y = zeros(N*N)

    rows = Vector{Int64}(undef,0)
    cols = Vector{Int64}(undef,0)
    vals = Vector{Float64}(undef,0)

    for i = 2:N-1
       
        for j = 2:N-1

            y[ixs[i,j]] = F[ixs[i,j]]
            
            append!(rows,ixs[i,j]); append!(cols,ixs[i,j]); append!(vals,4*exp(coef[ixs[i,j]])/h^2);
            append!(rows,ixs[i,j]); append!(cols,ixs[i-1,j]); append!(vals,-exp(coef[ixs[i,j]])/h^2);
            append!(rows,ixs[i,j]); append!(cols,ixs[i,j-1]); append!(vals,-exp(coef[ixs[i,j]])/h^2);
            append!(rows,ixs[i,j]); append!(cols,ixs[i+1,j]); append!(vals,-exp(coef[ixs[i,j]])/h^2);
            append!(rows,ixs[i,j]); append!(cols,ixs[i,j+1]); append!(vals,-exp(coef[ixs[i,j]])/h^2);

            append!(rows,ixs[i,j]); append!(cols,ixs[i+1,j]); append!(vals,-exp(coef[ixs[i,j]])*(coef[ixs[i+1,j]]/(4h*h) - coef[ixs[i-1,j]]/(4h*h)));
            append!(rows,ixs[i,j]); append!(cols,ixs[i-1,j]); append!(vals,+exp(coef[ixs[i,j]])*(coef[ixs[i+1,j]]/(4h*h) -coef[ixs[i-1,j]]/(4h*h)));
            append!(rows,ixs[i,j]); append!(cols,ixs[i,j+1]); append!(vals,-exp(coef[ixs[i,j]])*(coef[ixs[i,j+1]]/(4h*h) - coef[ixs[i,j-1]]/(4h*h)));
            append!(rows,ixs[i,j]); append!(cols,ixs[i,j-1]); append!(vals,+exp(coef[ixs[i,j]])*(coef[ixs[i,j+1]]/(4h*h) -coef[ixs[i,j-1]]/(4h*h)));


            # M[ixs[i,j],ixs[i,j]] += 4*coef[ixs[i,j]]/h^2
            # M[ixs[i,j],ixs[i-1,j]] += -coef[ixs[i,j]]/h^2
            # M[ixs[i,j],ixs[i,j-1]] += -coef[ixs[i,j]]/h^2
            # M[ixs[i,j],ixs[i+1,j]] += -coef[ixs[i,j]]/h^2
            # M[ixs[i,j],ixs[i,j+1]] += -coef[ixs[i,j]]/h^2


            # M[ixs[i,j],ixs[i+1,j]] += -(coef[ixs[i+1,j]]/(4h*h) - coef[ixs[i-1,j]]/(4h*h))
            # M[ixs[i,j],ixs[i-1,j]] += +(coef[ixs[i+1,j]]/(4h*h) -coef[ixs[i-1,j]]/(4h*h))
            # M[ixs[i,j],ixs[i,j+1]] += -(coef[ixs[i,j+1]]/(4h*h) - coef[ixs[i,j-1]]/(4h*h))
            # M[ixs[i,j],ixs[i,j-1]] += +(coef[ixs[i,j+1]]/(4h*h) -coef[ixs[i,j-1]]/(4h*h))


            # M[ixs[i,j],ixs[i,j]] += ((coef[ixs[i-1,j]]+coef[ixs[i,j]])/2 + (coef[ixs[i,j]]+coef[ixs[i+1,j]])/2 + (coef[ixs[i,j]]+coef[ixs[i,j+1]])/2 + (coef[ixs[i,j-1]]+coef[ixs[i,j]])/2)/h^2
            # M[ixs[i,j],ixs[i+1,j]] += -(coef[ixs[i,j]]/(2h*h) + coef[ixs[i+1,j]]/(2h*h))
            # M[ixs[i,j],ixs[i-1,j]] += -(coef[ixs[i,j]]/(2h*h) + coef[ixs[i+1,j]]/(2h*h))
            # M[ixs[i,j],ixs[i,j+1]] += -(coef[ixs[i,j]]/(2h*h) + coef[ixs[i,j+1]]/(2h*h))
            # M[ixs[i,j],ixs[i,j-1]] += -(coef[ixs[i,j]]/(2h*h) + coef[ixs[i,j+1]]/(2h*h))



            # M[ixs[i,j],ixs[i+1,j]] += -(coef[ixs[i+1,j]]/(h*h) - coef[ixs[i,j]]/(h*h))
            # M[ixs[i,j],ixs[i,j]] += coef[ixs[i+1,j]]/(h*h) -coef[ixs[i,j]]/(h*h)
            # M[ixs[i,j],ixs[i,j+1]] += -(coef[ixs[i,j+1]]/(h*h) - coef[ixs[i,j]]/(h*h))
            # M[ixs[i,j],ixs[i,j]] += coef[ixs[i,j+1]]/(h*h) -coef[ixs[i,j]]/(h*h)


            # M[ixs[i,j],ixs[i,j]] += -(coef[ixs[i,j]]/(h*h) - coef[ixs[i-1,j]]/(h*h))
            # M[ixs[i,j],ixs[i-1,j]] += coef[ixs[i,j]]/(h*h) -coef[ixs[i-1,j]]/(h*h)
            # M[ixs[i,j],ixs[i,j]] += -(coef[ixs[i,j]]/(h*h) - coef[ixs[i,j-1]]/(h*h))
            # M[ixs[i,j],ixs[i,j-1]] += coef[ixs[i,j]]/(h*h) -coef[ixs[i,j-1]]/(h*h)
  
        end
    end

    
    #B = Diagonal(zeros(N^2))
    #Boff = I(N^2)

    #Vasen reuna.
    # for i = 2:N-1
  
    #     # M[ixs[i,1],ixs[i,1]] = 1
    #     # Boff[ixs[i,1],ixs[i,1]] = 0
    #     # B[ixs[i,1],ixs[i,1]] = 1.0
    #     # y[ixs[i,1]] = 0
    #     # append!(rows,ixs[i,1]); append!(cols,ixs[i,1]); append!(vals,1.0);

    #     # exp(k)*x_{2,i} - 2*h*g_{1,i}  = exp(k)*x{0,i}
    #     # x{0,i}*-exp(coef[ixs[1,i]])/h^2 = (x_{2,i} - 2*h*g_{1,i})*-exp(coef[ixs[1,i]])/(h^2*exp(k)) 

    #     gv = 500.0

    #     append!(rows,ixs[i,1]); append!(cols,ixs[i,1]); append!(vals,4*exp(coef[ixs[i,1]])/h^2);
    #     append!(rows,ixs[i,1]); append!(cols,ixs[i,1+1]); append!(vals,-exp(coef[ixs[i,1]])/h^2);
    #     append!(rows,ixs[i,1]); append!(cols,ixs[i,1+1]); append!(vals,-exp(coef[ixs[i,1]])/h^2);
    #     append!(rows,ixs[i,1]); append!(cols,ixs[i-1,1]); append!(vals,-exp(coef[ixs[i,1]])/h^2);
    #     append!(rows,ixs[i,1]); append!(cols,ixs[i+1,1]); append!(vals,-exp(coef[ixs[i,1]])/h^2);
    #     ##append!(rows,ixs[1,i]); append!(cols,ixs[1-1,i]); append!(vals,-coef[ixs[1,i]]/h^2);

    #     ##append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,-(coef[ixs[1+1,i]]/(4h*h) - coef[ixs[1-1,i]]/(4h*h)));
    #     ##append!(rows,ixs[1,i]); append!(cols,ixs[1-1,i]); append!(vals,+(coef[ixs[1+1,j]]/(4h*h) -coef[ixs[1-1,i]]/(4h*h)));
    #     #append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,-(coef[ixs[1+1,i]]/(2h*h) - coef[ixs[1,i]]/(2h*h)));
    #     #append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,+(coef[ixs[1+1,i]]/(2h*h) - coef[ixs[1,i]]/(2h*h)));
    #     append!(rows,ixs[i,1]); append!(cols,ixs[i+1,1]); append!(vals,-exp(coef[ixs[i,1]])*(coef[ixs[i+1,1]]/(4h*h) - coef[ixs[i-1,1]]/(4h*h)));
    #     append!(rows,ixs[i,1]); append!(cols,ixs[i-1,1]); append!(vals,+exp(coef[ixs[i,1]])*(coef[ixs[i+1,1]]/(4h*h) - coef[ixs[i-1,1]]/(4h*h)));

    #     y[ixs[i,1]] =  F[ixs[i,1]] + 2*gv/h


    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i,j]); append!(vals,4*exp(coef[ixs[i,j]])/h^2);
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i-1,j]); append!(vals,-exp(coef[ixs[i,j]])/h^2);
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i,j-1]); append!(vals,-exp(coef[ixs[i,j]])/h^2);
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i+1,j]); append!(vals,-exp(coef[ixs[i,j]])/h^2);
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i,j+1]); append!(vals,-exp(coef[ixs[i,j]])/h^2);

    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i+1,j]); append!(vals,-exp(coef[ixs[i,j]])*(coef[ixs[i+1,j]]/(4h*h) - coef[ixs[i-1,j]]/(4h*h)));
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i-1,j]); append!(vals,+exp(coef[ixs[i,j]])*(coef[ixs[i+1,j]]/(4h*h) -coef[ixs[i-1,j]]/(4h*h)));
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i,j+1]); append!(vals,-exp(coef[ixs[i,j]])*(coef[ixs[i,j+1]]/(4h*h) - coef[ixs[i,j-1]]/(4h*h)));
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i,j-1]); append!(vals,+exp(coef[ixs[i,j]])*(coef[ixs[i,j+1]]/(4h*h) -coef[ixs[i,j-1]]/(4h*h)));
      

    # end

    # #Vasen ylänurkka.
    # for i = 1:1

    #     # exp(k)*(x_{1,2}-x_{1,0})/(2h) = g_{1,1}
    #     # exp(k)*(x_{2,1}-x_{0,1})/(2h) = g_{1,1}

    #     # exp(k)*x_{2,1} - 2*h*g_{1,1}  = exp(k)*x{0,1}


    #     gv = 500.0

    #     append!(rows,ixs[i,1]); append!(cols,ixs[i,1]); append!(vals,4*exp(coef[ixs[i,1]])/h^2);
    #     append!(rows,ixs[i,1]); append!(cols,ixs[i,1+1]); append!(vals,-exp(coef[ixs[i,1]])/h^2);
    #     append!(rows,ixs[i,1]); append!(cols,ixs[i,1+1]); append!(vals,-exp(coef[ixs[i,1]])/h^2);
    #     append!(rows,ixs[i,1]); append!(cols,ixs[i+1,1]); append!(vals,-exp(coef[ixs[i,1]])/h^2);
    #     append!(rows,ixs[i,1]); append!(cols,ixs[i+1,1]); append!(vals,-exp(coef[ixs[i,1]])/h^2);
    #     ##append!(rows,ixs[1,i]); append!(cols,ixs[1-1,i]); append!(vals,-coef[ixs[1,i]]/h^2);

    #     ##append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,-(coef[ixs[1+1,i]]/(4h*h) - coef[ixs[1-1,i]]/(4h*h)));
    #     ##append!(rows,ixs[1,i]); append!(cols,ixs[1-1,i]); append!(vals,+(coef[ixs[1+1,j]]/(4h*h) -coef[ixs[1-1,i]]/(4h*h)));
    #     # append!(rows,ixs[i,1]); append!(cols,ixs[1+1,i]); append!(vals,-(coef[ixs[1+1,i]]/(2h*h) - coef[ixs[1,i]]/(2h*h)));
    #     # append!(rows,ixs[i,1]); append!(cols,ixs[1+1,i]); append!(vals,+(coef[ixs[1+1,i]]/(2h*h) - coef[ixs[1,i]]/(2h*h)));
    #     # append!(rows,ixs[i,1]); append!(cols,ixs[1,i+1]); append!(vals,-exp(coef[ixs[1,i]])*(coef[ixs[1,i+1]]/(4h*h) - coef[ixs[1,i-1]]/(4h*h)));
    #     # append!(rows,ixs[i,1]); append!(cols,ixs[1,i-1]); append!(vals,+exp(coef[ixs[1,i]])*(coef[ixs[1,i+1]]/(4h*h) - coef[ixs[1,i-1]]/(4h*h)));

    #     y[ixs[i,1]] =  F[ixs[i,1]] + 4*gv/h

    # end

    # #Yläreuna.
    # for i = 2:N-1

    #     # exp(k)*x_{2,i} - 2*h*g_{1,i}  = exp(k)*x{0,i}
    #     # x{0,i}*-exp(coef[ixs[1,i]])/h^2 = (x_{2,i} - 2*h*g_{1,i})*-exp(coef[ixs[1,i]])/h^2 

    #     gv = 0.0

    #     # append!(rows,ixs[1,i]); append!(cols,ixs[1,i]); append!(vals,-1);
    #     # append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,1);
    #     # y[ixs[1,i]] = 0

    #     append!(rows,ixs[1,i]); append!(cols,ixs[1,i]); append!(vals,4*exp(coef[ixs[1,i]])/h^2);
    #     append!(rows,ixs[1,i]); append!(cols,ixs[1,i+1]); append!(vals,-exp(coef[ixs[1,i]])/h^2);
    #     append!(rows,ixs[1,i]); append!(cols,ixs[1,i-1]); append!(vals,-exp(coef[ixs[1,i]])/h^2);
    #     append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,-exp(coef[ixs[1,i]])/h^2);
    #     append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,-exp(coef[ixs[1,i]])/h^2);
    #     ##append!(rows,ixs[1,i]); append!(cols,ixs[1-1,i]); append!(vals,-coef[ixs[1,i]]/h^2);

    #     ##append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,-(coef[ixs[1+1,i]]/(4h*h) - coef[ixs[1-1,i]]/(4h*h)));
    #     ##append!(rows,ixs[1,i]); append!(cols,ixs[1-1,i]); append!(vals,+(coef[ixs[1+1,j]]/(4h*h) -coef[ixs[1-1,i]]/(4h*h)));
    #     #append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,-(coef[ixs[1+1,i]]/(2h*h) - coef[ixs[1,i]]/(2h*h)));
    #     #append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,+(coef[ixs[1+1,i]]/(2h*h) - coef[ixs[1,i]]/(2h*h)));
    #     append!(rows,ixs[1,i]); append!(cols,ixs[1,i+1]); append!(vals,-exp(coef[ixs[1,i]])*(coef[ixs[1,i+1]]/(4h*h) - coef[ixs[1,i-1]]/(4h*h)));
    #     append!(rows,ixs[1,i]); append!(cols,ixs[1,i-1]); append!(vals,+exp(coef[ixs[1,i]])*(coef[ixs[1,i+1]]/(4h*h) - coef[ixs[1,i-1]]/(4h*h)));

    #     y[ixs[1,i]] =  F[ixs[1,i]] + 2*gv/h


    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i,j]); append!(vals,4*exp(coef[ixs[i,j]])/h^2);
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i-1,j]); append!(vals,-exp(coef[ixs[i,j]])/h^2);
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i,j-1]); append!(vals,-exp(coef[ixs[i,j]])/h^2);
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i+1,j]); append!(vals,-exp(coef[ixs[i,j]])/h^2);
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i,j+1]); append!(vals,-exp(coef[ixs[i,j]])/h^2);

    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i+1,j]); append!(vals,-exp(coef[ixs[i,j]])*(coef[ixs[i+1,j]]/(4h*h) - coef[ixs[i-1,j]]/(4h*h)));
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i-1,j]); append!(vals,+exp(coef[ixs[i,j]])*(coef[ixs[i+1,j]]/(4h*h) -coef[ixs[i-1,j]]/(4h*h)));
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i,j+1]); append!(vals,-exp(coef[ixs[i,j]])*(coef[ixs[i,j+1]]/(4h*h) - coef[ixs[i,j-1]]/(4h*h)));
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i,j-1]); append!(vals,+exp(coef[ixs[i,j]])*(coef[ixs[i,j+1]]/(4h*h) -coef[ixs[i,j-1]]/(4h*h)));
      


    # end

    # #Oikea reuna.
    # for i = 2:N-1
  
    #     # M[ixs[i,N],ixs[i,N]] = 1
    #     # Boff[ixs[i,N],ixs[i,N]] = 0
    #     # B[ixs[i,N],ixs[i,N]] = 1.0
    #     #y[ixs[i,N]] = 0
    #     #append!(rows,ixs[i,N]); append!(cols,ixs[i,N]); append!(vals,1.0);


    #     gv = 0.0

    #     append!(rows,ixs[i,N]); append!(cols,ixs[i,N]); append!(vals,4*exp(coef[ixs[i,N]])/h^2);
    #     append!(rows,ixs[i,N]); append!(cols,ixs[i,N-1]); append!(vals,-exp(coef[ixs[i,N]])/h^2);
    #     append!(rows,ixs[i,N]); append!(cols,ixs[i,N-1]); append!(vals,-exp(coef[ixs[i,N]])/h^2);
    #     append!(rows,ixs[i,N]); append!(cols,ixs[i+1,N]); append!(vals,-exp(coef[ixs[i,N]])/h^2);
    #     append!(rows,ixs[i,N]); append!(cols,ixs[i-1,N]); append!(vals,-exp(coef[ixs[i,N]])/h^2);
    #     ##append!(rows,ixs[1,i]); append!(cols,ixs[1-1,i]); append!(vals,-coef[ixs[1,i]]/h^2);

    #     ##append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,-(coef[ixs[1+1,i]]/(4h*h) - coef[ixs[1-1,i]]/(4h*h)));
    #     ##append!(rows,ixs[1,i]); append!(cols,ixs[1-1,i]); append!(vals,+(coef[ixs[1+1,j]]/(4h*h) -coef[ixs[1-1,i]]/(4h*h)));
    #     #append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,-(coef[ixs[1+1,i]]/(2h*h) - coef[ixs[1,i]]/(2h*h)));
    #     #append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,+(coef[ixs[1+1,i]]/(2h*h) - coef[ixs[1,i]]/(2h*h)));
    #     append!(rows,ixs[i,N]); append!(cols,ixs[i+1,N]); append!(vals,-exp(coef[ixs[i,N]])*(coef[ixs[i+1,N]]/(4h*h) - coef[ixs[i-1,N]]/(4h*h)));
    #     append!(rows,ixs[i,N]); append!(cols,ixs[i-1,N]); append!(vals,+exp(coef[ixs[i,N]])*(coef[ixs[i+1,N]]/(4h*h) - coef[ixs[i-1,N]]/(4h*h)));

    #     y[ixs[i,N]] =  F[ixs[i,N]] + 2*gv/h


    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i,j]); append!(vals,4*exp(coef[ixs[i,j]])/h^2);
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i-1,j]); append!(vals,-exp(coef[ixs[i,j]])/h^2);
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i,j-1]); append!(vals,-exp(coef[ixs[i,j]])/h^2);
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i+1,j]); append!(vals,-exp(coef[ixs[i,j]])/h^2);
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i,j+1]); append!(vals,-exp(coef[ixs[i,j]])/h^2);

    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i+1,j]); append!(vals,-exp(coef[ixs[i,j]])*(coef[ixs[i+1,j]]/(4h*h) - coef[ixs[i-1,j]]/(4h*h)));
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i-1,j]); append!(vals,+exp(coef[ixs[i,j]])*(coef[ixs[i+1,j]]/(4h*h) -coef[ixs[i-1,j]]/(4h*h)));
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i,j+1]); append!(vals,-exp(coef[ixs[i,j]])*(coef[ixs[i,j+1]]/(4h*h) - coef[ixs[i,j-1]]/(4h*h)));
    #     # append!(rows,ixs[i,j]); append!(cols,ixs[i,j-1]); append!(vals,+exp(coef[ixs[i,j]])*(coef[ixs[i,j+1]]/(4h*h) -coef[ixs[i,j-1]]/(4h*h)));

     
    # end 

    # #Oikea ylänurkka.
    # for i = 1:1

    #     # exp(k)*(x_{1,N+1}-x_{1,N-1})/(2h) = g_{1,N}
    #     # exp(k)*(x_{2,N}-x_{0,N})/(2h) = g_{1,N}

    #     # exp(k)*x_{i,N} - 2*h*g_{i,N}  = exp(k)*x{0,i}
    #     # x{0,i}*-exp(coef[ixs[1,i]])/h^2 = (x_{2,i} - 2*h*g_{1,i})*-exp(coef[ixs[1,i]])/h^2 

    #     gv = 0.0

    #     append!(rows,ixs[i,N]); append!(cols,ixs[i,N]); append!(vals,4*exp(coef[ixs[i,N]])/h^2);
    #     append!(rows,ixs[i,N]); append!(cols,ixs[i,N-1]); append!(vals,-exp(coef[ixs[i,N]])/h^2);
    #     append!(rows,ixs[i,N]); append!(cols,ixs[i,N-1]); append!(vals,-exp(coef[ixs[i,N]])/h^2);
    #     append!(rows,ixs[i,N]); append!(cols,ixs[i+1,N]); append!(vals,-exp(coef[ixs[i,N]])/h^2);
    #     append!(rows,ixs[i,N]); append!(cols,ixs[i+1,N]); append!(vals,-exp(coef[ixs[i,N]])/h^2);
    #     ##append!(rows,ixs[1,i]); append!(cols,ixs[1-1,i]); append!(vals,-coef[ixs[1,i]]/h^2);

    #     ##append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,-(coef[ixs[1+1,i]]/(4h*h) - coef[ixs[1-1,i]]/(4h*h)));
    #     ##append!(rows,ixs[1,i]); append!(cols,ixs[1-1,i]); append!(vals,+(coef[ixs[1+1,j]]/(4h*h) -coef[ixs[1-1,i]]/(4h*h)));
    #     #append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,-(coef[ixs[1+1,i]]/(2h*h) - coef[ixs[1,i]]/(2h*h)));
    #     #append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,+(coef[ixs[1+1,i]]/(2h*h) - coef[ixs[1,i]]/(2h*h)));
    #     #append!(rows,ixs[i,N]); append!(cols,ixs[1,i+1]); append!(vals,-exp(coef[ixs[1,i]])*(coef[ixs[1,i+1]]/(4h*h) - coef[ixs[1,i-1]]/(4h*h)));
    #     #append!(rows,ixs[i,N]); append!(cols,ixs[1,i-1]); append!(vals,+exp(coef[ixs[1,i]])*(coef[ixs[1,i+1]]/(4h*h) - coef[ixs[1,i-1]]/(4h*h)));

    #     y[ixs[i,N]] =  F[ixs[i,N]] + 4*gv/h

    # end
    
    # #Alareuna.
    # for i = 1:N
  
    #     # M[ixs[N,i],ixs[N,i]] = 1
    #     #Boff[ixs[N,i],ixs[N,i]] = 0
    #     #B[ixs[N,i],ixs[N,i]] = 1.0
    #     y[ixs[N,i]] = 100

    #     append!(rows,ixs[N,i]); append!(cols,ixs[N,i]); append!(vals,1.0);

    # end
    
    
   

    boundary = Set([ixs[1,i] for i in 1:N])
    push!(boundary, [ixs[N,i] for i in 1:N]...)
    push!(boundary, [ixs[i,1] for i in 1:N]...)
    push!(boundary, [ixs[i,N] for i in 1:N]...)

    for b in boundary
        append!(rows,ixs[b]); append!(cols,ixs[b]); append!(vals,1.0);
        y[ixs[b]] = 0
    end

    #= for i = 1:N

        # append!(rows,ixs[i,1]); append!(cols,ixs[i,1]); append!(vals,1.0);
        # append!(rows,ixs[i,N]); append!(cols,ixs[i,N]); append!(vals,1.0);
        # append!(rows,ixs[1,i]); append!(cols,ixs[1,i]); append!(vals,1.0);
        # append!(rows,ixs[N,i]); append!(cols,ixs[N,i]); append!(vals,1.0);

        # M[ixs[i,1],ixs[i,1]] = 1
        # M[ixs[i,N],ixs[i,N]] = 1

        # M[ixs[1,i],ixs[1,i]] = 1
        # M[ixs[N,i],ixs[N,i]] = 1

        Boff[ixs[i,1],ixs[i,1]] = 0
        Boff[ixs[i,N],ixs[i,N]] = 0
        Boff[ixs[1,i],ixs[1,i]] = 0
        Boff[ixs[N,i],ixs[N,i]] = 0

        B[ixs[i,1],ixs[i,1]] = 1.0
        B[ixs[i,N],ixs[i,N]] = 1.0
        B[ixs[1,i],ixs[1,i]] = 1.0
        B[ixs[N,i],ixs[N,i]] = 1.0
        
        
        y[ixs[i,1]] = 0
        y[ixs[i,N]] = 0
        y[ixs[1,i]] = 0
        y[ixs[N,i]] = 0

    end =#
    
    #return M,y
    #return Boff*M + B,y

    M = sparse(rows,cols,vals,N^2,N^2)
    return M,y
end




# function equderix(co,f,x)

#     #=
#     N = Int64(sqrt(length(x)))
#     ixa = CartesianIndices( (1:N,1:N) )

#     if xindex ∉ NZR[eqindex]
#         return nothing

#     else
#         M1 = solution(x,f)[1]
#     end
#     =#
#     M,_ = solution(co,f)
#     return M
#     #return Diagonal(co.^2)


# end


function equderipar(X,Y,coef,F,x)

    N = Int64(sqrt(length(coef)))
    ##M = zeros(N^2,N^2)
    ixs = LinearIndices((1:N,1:N))
    h = X[1,2]-X[1,1]

    rows = Vector{Int64}(undef,0)
    cols = Vector{Int64}(undef,0)
    vals = Vector{Float64}(undef,0)

    for i = 2:N-1
       
        for j = 2:N-1
            
            # append!(rows,ixs[i,j]); append!(cols,ixs[i,j]); append!(vals,4*coef[ixs[i,j]]/h^2);
            # append!(rows,ixs[i,j]); append!(cols,ixs[i-1,j]); append!(vals,-coef[ixs[i,j]]/h^2);
            # append!(rows,ixs[i,j]); append!(cols,ixs[i,j-1]); append!(vals,-coef[ixs[i,j]]/h^2);
            # append!(rows,ixs[i,j]); append!(cols,ixs[i+1,j]); append!(vals,-coef[ixs[i,j]]/h^2);
            # append!(rows,ixs[i,j]); append!(cols,ixs[i,j+1]); append!(vals,-coef[ixs[i,j]]/h^2);

            # append!(rows,ixs[i,j]); append!(cols,ixs[i+1,j]); append!(vals,-(coef[ixs[i+1,j]]/(4h*h) - coef[ixs[i-1,j]]/(4h*h)));
            # append!(rows,ixs[i,j]); append!(cols,ixs[i-1,j]); append!(vals,+(coef[ixs[i+1,j]]/(4h*h) -coef[ixs[i-1,j]]/(4h*h)));
            # append!(rows,ixs[i,j]); append!(cols,ixs[i,j+1]); append!(vals,-(coef[ixs[i,j+1]]/(4h*h) - coef[ixs[i,j-1]]/(4h*h)));
            # append!(rows,ixs[i,j]); append!(cols,ixs[i,j-1]); append!(vals,+(coef[ixs[i,j+1]]/(4h*h) -coef[ixs[i,j-1]]/(4h*h)));


            append!(rows,ixs[i,j]); append!(cols,ixs[i,j]); append!(vals, 4*x[ixs[i,j]]/h^2)
            append!(rows,ixs[i,j]); append!(cols,ixs[i,j]); append!(vals, -x[ixs[i-1,j]]/h^2)
            append!(rows,ixs[i,j]); append!(cols,ixs[i,j]); append!(vals, -x[ixs[i,j-1]]/h^2)
            append!(rows,ixs[i,j]); append!(cols,ixs[i,j]); append!(vals, -x[ixs[i+1,j]]/h^2)
            append!(rows,ixs[i,j]); append!(cols,ixs[i,j]); append!(vals, -x[ixs[i,j+1]]/h^2)

            append!(rows,ixs[i,j]); append!(cols,ixs[i+1,j]); append!(vals, -(x[ixs[i+1,j]]/(4*h*h)))
            append!(rows,ixs[i,j]); append!(cols,ixs[i-1,j]); append!(vals, +(x[ixs[i+1,j]]/(4*h*h)))
            append!(rows,ixs[i,j]); append!(cols,ixs[i+1,j]); append!(vals, +(x[ixs[i-1,j]]/(4*h*h)))
            append!(rows,ixs[i,j]); append!(cols,ixs[i-1,j]); append!(vals, -(x[ixs[i-1,j]]/(4*h*h)))

            append!(rows,ixs[i,j]); append!(cols,ixs[i,j+1]); append!(vals, -(x[ixs[i,j+1]]/(4*h*h)))
            append!(rows,ixs[i,j]); append!(cols,ixs[i,j-1]); append!(vals, +(x[ixs[i,j+1]]/(4*h*h)))
            append!(rows,ixs[i,j]); append!(cols,ixs[i,j+1]); append!(vals, +(x[ixs[i,j-1]]/(4*h*h)))
            append!(rows,ixs[i,j]); append!(cols,ixs[i,j-1]); append!(vals, -(x[ixs[i,j-1]]/(4*h*h)))

            # M[ixs[i,j],ixs[i,j]] += 4*x[ixs[i,j]]/h^2
            # M[ixs[i,j],ixs[i,j]] += -x[ixs[i-1,j]]/h^2
            # M[ixs[i,j],ixs[i,j]] += -x[ixs[i,j-1]]/h^2
            # M[ixs[i,j],ixs[i,j]] += -x[ixs[i+1,j]]/h^2
            # M[ixs[i,j],ixs[i,j]] += -x[ixs[i,j+1]]/h^2


            # M[ixs[i,j],ixs[i+1,j]] += -(x[ixs[i+1,j]]/(4*h*h))
            # M[ixs[i,j],ixs[i-1,j]] += +(x[ixs[i+1,j]]/(4*h*h))
            # M[ixs[i,j],ixs[i+1,j]] += +(x[ixs[i-1,j]]/(4*h*h))
            # M[ixs[i,j],ixs[i-1,j]] += -(x[ixs[i-1,j]]/(4*h*h))

            # M[ixs[i,j],ixs[i,j+1]] += -(x[ixs[i,j+1]]/(4*h*h))
            # M[ixs[i,j],ixs[i,j-1]] += +(x[ixs[i,j+1]]/(4*h*h))
            # M[ixs[i,j],ixs[i,j+1]] += +(x[ixs[i,j-1]]/(4*h*h))
            # M[ixs[i,j],ixs[i,j-1]] += -(x[ixs[i,j-1]]/(4*h*h))

            # M[ixs[i,j],ixs[i+1,j]] += -(coef[ixs[i+1,j]]/(4h*h) - coef[ixs[i-1,j]]/(4h*h))
            # M[ixs[i,j],ixs[i-1,j]] += +(coef[ixs[i+1,j]]/(4h*h) -coef[ixs[i-1,j]]/(4h*h))
            # M[ixs[i,j],ixs[i,j+1]] += -(coef[ixs[i,j+1]]/(4h*h) - coef[ixs[i,j-1]]/(4h*h))
            # M[ixs[i,j],ixs[i,j-1]] += +(coef[ixs[i,j+1]]/(4h*h) -coef[ixs[i,j-1]]/(4h*h))


  
        end
    end

    #M = sparse(rows,cols,vals,N^2,N^2)
    #B = spdiagm(N^2,N^2,0=>zeros(N^2))
    #B = Diagonal(zeros(N^2))
    #Boff = I(N^2)
    #for i = 1:N

        # append!(rows,ixs[i,1]); append!(cols,ixs[i,1]); append!(vals,1.0);
        # append!(rows,ixs[i,N]); append!(cols,ixs[i,N]); append!(vals,1.0);
        # append!(rows,ixs[1,i]); append!(cols,ixs[1,i]); append!(vals,1.0);
        # append!(rows,ixs[N,i]); append!(cols,ixs[N,i]); append!(vals,1.0);

        # M[ixs[i,1],ixs[i,1]] = 1
        # M[ixs[i,N],ixs[i,N]] = 1

        # M[ixs[1,i],ixs[1,i]] = 1
        # M[ixs[N,i],ixs[N,i]] = 1

        # Boff[ixs[i,1],ixs[i,1]] = 0
        # Boff[ixs[i,N],ixs[i,N]] = 0
        # Boff[ixs[1,i],ixs[1,i]] = 0
        # Boff[ixs[N,i],ixs[N,i]] = 0

        # B[ixs[i,1],ixs[i,1]] = 1.0
        # B[ixs[i,N],ixs[i,N]] = 1.0
        # B[ixs[1,i],ixs[1,i]] = 1.0
        # B[ixs[N,i],ixs[N,i]] = 1.0
        
        

    #end
    
    #return M,y
    M = sparse(rows,cols,vals,N^2,N^2)
    return M# Boff*M + B


    #return Diagonal(x.*2.0.*co)


end
 

function equderiparexp(X,Y,coef,F,x)

    N = Int64(sqrt(length(coef)))
    M = 1 #zeros(N^2,N^2)
    ixs = LinearIndices((1:N,1:N))
    #h = 1/(N-1)
    h = X[1,2]-X[1,1]

    rows = Vector{Int64}(undef,0)
    cols = Vector{Int64}(undef,0)
    vals = Vector{Float64}(undef,0)

    for i = 2:N-1
       
        for j = 2:N-1

            # M[ixs[i,j],ixs[i,j]] += 4*x[ixs[i,j]]*exp(coef[ixs[i,j]])/h^2
            # M[ixs[i,j],ixs[i,j]] += -x[ixs[i-1,j]]*exp(coef[ixs[i,j]])/h^2
            # M[ixs[i,j],ixs[i,j]] += -x[ixs[i,j-1]]*exp(coef[ixs[i,j]])/h^2
            # M[ixs[i,j],ixs[i,j]] += -x[ixs[i+1,j]]*exp(coef[ixs[i,j]])/h^2
            # M[ixs[i,j],ixs[i,j]] += -x[ixs[i,j+1]]*exp(coef[ixs[i,j]])/h^2


            # M[ixs[i,j],ixs[i+1,j]] += -exp(coef[ixs[i,j]])*(x[ixs[i+1,j]]/(4*h*h))
            # M[ixs[i,j],ixs[i-1,j]] += +exp(coef[ixs[i,j]])*(x[ixs[i+1,j]]/(4*h*h))
            # M[ixs[i,j],ixs[i+1,j]] += +exp(coef[ixs[i,j]])*(x[ixs[i-1,j]]/(4*h*h))
            # M[ixs[i,j],ixs[i-1,j]] += -exp(coef[ixs[i,j]])*(x[ixs[i-1,j]]/(4*h*h))

            # M[ixs[i,j],ixs[i,j+1]] += -exp(coef[ixs[i,j]])*(x[ixs[i,j+1]]/(4*h*h))
            # M[ixs[i,j],ixs[i,j-1]] += +exp(coef[ixs[i,j]])*(x[ixs[i,j+1]]/(4*h*h))
            # M[ixs[i,j],ixs[i,j+1]] += +exp(coef[ixs[i,j]])*(x[ixs[i,j-1]]/(4*h*h))
            # M[ixs[i,j],ixs[i,j-1]] += -exp(coef[ixs[i,j]])*(x[ixs[i,j-1]]/(4*h*h))

            # M[ixs[i,j],ixs[i,j]] += -exp(coef[ixs[i,j]])*(coef[ixs[i+1,j]]/(4h*h) - coef[ixs[i-1,j]]/(4h*h))*x[ixs[i+1,j]]
            # M[ixs[i,j],ixs[i,j]] += +exp(coef[ixs[i,j]])*(coef[ixs[i+1,j]]/(4h*h) - coef[ixs[i-1,j]]/(4h*h))*x[ixs[i-1,j]]
            # M[ixs[i,j],ixs[i,j]] += -exp(coef[ixs[i,j]])*(coef[ixs[i,j+1]]/(4h*h) - coef[ixs[i,j-1]]/(4h*h))*x[ixs[i,j+1]]
            # M[ixs[i,j],ixs[i,j]] += +exp(coef[ixs[i,j]])*(coef[ixs[i,j+1]]/(4h*h) - coef[ixs[i,j-1]]/(4h*h))*x[ixs[i,j-1]]


            append!(rows,ixs[i,j]);  append!(cols,ixs[i,j]);  append!(vals, 4*x[ixs[i,j]]*exp(coef[ixs[i,j]])/h^2); 
            append!(rows,ixs[i,j]);  append!(cols,ixs[i,j]);  append!(vals, -x[ixs[i-1,j]]*exp(coef[ixs[i,j]])/h^2); 
            append!(rows,ixs[i,j]);  append!(cols,ixs[i,j]);  append!(vals, -x[ixs[i,j-1]]*exp(coef[ixs[i,j]])/h^2); 
            append!(rows,ixs[i,j]);  append!(cols,ixs[i,j]);  append!(vals, -x[ixs[i+1,j]]*exp(coef[ixs[i,j]])/h^2); 
            append!(rows,ixs[i,j]);  append!(cols,ixs[i,j]);  append!(vals, -x[ixs[i,j+1]]*exp(coef[ixs[i,j]])/h^2); 

            append!(rows,ixs[i,j]);  append!(cols,ixs[i+1,j]);  append!(vals, -exp(coef[ixs[i,j]])*(x[ixs[i+1,j]]/(4*h*h))); 
            append!(rows,ixs[i,j]);  append!(cols,ixs[i-1,j]);  append!(vals, +exp(coef[ixs[i,j]])*(x[ixs[i+1,j]]/(4*h*h))); 
            append!(rows,ixs[i,j]);  append!(cols,ixs[i+1,j]);  append!(vals, +exp(coef[ixs[i,j]])*(x[ixs[i-1,j]]/(4*h*h))); 
            append!(rows,ixs[i,j]);  append!(cols,ixs[i-1,j]);  append!(vals, -exp(coef[ixs[i,j]])*(x[ixs[i-1,j]]/(4*h*h))); 

            append!(rows,ixs[i,j]);  append!(cols,ixs[i,j+1]);  append!(vals, -exp(coef[ixs[i,j]])*(x[ixs[i,j+1]]/(4*h*h))); 
            append!(rows,ixs[i,j]);  append!(cols,ixs[i,j-1]);  append!(vals, +exp(coef[ixs[i,j]])*(x[ixs[i,j+1]]/(4*h*h))); 
            append!(rows,ixs[i,j]);  append!(cols,ixs[i,j+1]);  append!(vals, +exp(coef[ixs[i,j]])*(x[ixs[i,j-1]]/(4*h*h))); 
            append!(rows,ixs[i,j]);  append!(cols,ixs[i,j-1]);  append!(vals, -exp(coef[ixs[i,j]])*(x[ixs[i,j-1]]/(4*h*h))); 

            append!(rows,ixs[i,j]);  append!(cols,ixs[i,j]);  append!(vals, -exp(coef[ixs[i,j]])*(coef[ixs[i+1,j]]/(4h*h) - coef[ixs[i-1,j]]/(4h*h))*x[ixs[i+1,j]]); 
            append!(rows,ixs[i,j]);  append!(cols,ixs[i,j]);  append!(vals, +exp(coef[ixs[i,j]])*(coef[ixs[i+1,j]]/(4h*h) - coef[ixs[i-1,j]]/(4h*h))*x[ixs[i-1,j]]); 
            append!(rows,ixs[i,j]);  append!(cols,ixs[i,j]);  append!(vals, -exp(coef[ixs[i,j]])*(coef[ixs[i,j+1]]/(4h*h) - coef[ixs[i,j-1]]/(4h*h))*x[ixs[i,j+1]]); 
            append!(rows,ixs[i,j]);  append!(cols,ixs[i,j]);  append!(vals, +exp(coef[ixs[i,j]])*(coef[ixs[i,j+1]]/(4h*h) - coef[ixs[i,j-1]]/(4h*h))*x[ixs[i,j-1]]); 

  
            
            # append!(rows,ixs[i,j]); append!(cols,ixs[i,j]); append!(vals,4*exp(coef[ixs[i,j]])/h^2);
            # append!(rows,ixs[i,j]); append!(cols,ixs[i-1,j]); append!(vals,-exp(coef[ixs[i,j]])/h^2);
            # append!(rows,ixs[i,j]); append!(cols,ixs[i,j-1]); append!(vals,-exp(coef[ixs[i,j]])/h^2);
            # append!(rows,ixs[i,j]); append!(cols,ixs[i+1,j]); append!(vals,-exp(coef[ixs[i,j]])/h^2);
            # append!(rows,ixs[i,j]); append!(cols,ixs[i,j+1]); append!(vals,-exp(coef[ixs[i,j]])/h^2);

            # append!(rows,ixs[i,j]); append!(cols,ixs[i+1,j]); append!(vals,-exp(coef[ixs[i,j]])*(coef[ixs[i+1,j]]/(4h*h) - coef[ixs[i-1,j]]/(4h*h)));
            # append!(rows,ixs[i,j]); append!(cols,ixs[i-1,j]); append!(vals,+exp(coef[ixs[i,j]])*(coef[ixs[i+1,j]]/(4h*h) -coef[ixs[i-1,j]]/(4h*h)));
            # append!(rows,ixs[i,j]); append!(cols,ixs[i,j+1]); append!(vals,-exp(coef[ixs[i,j]])*(coef[ixs[i,j+1]]/(4h*h) - coef[ixs[i,j-1]]/(4h*h)));
            # append!(rows,ixs[i,j]); append!(cols,ixs[i,j-1]); append!(vals,+exp(coef[ixs[i,j]])*(coef[ixs[i,j+1]]/(4h*h) -coef[ixs[i,j-1]]/(4h*h)));



        end
    end

    #B = Diagonal(zeros(N^2))
    #Boff = I(N^2)

    #Vasen reuna.
    # for i = 2:N-1
  
    #     # M[ixs[i,1],ixs[i,1]] = 1
    #     # Boff[ixs[i,1],ixs[i,1]] = 0
    #     # B[ixs[i,1],ixs[i,1]] = 1.0
    #     # y[ixs[i,1]] = 0
    #     # append!(rows,ixs[i,1]); append!(cols,ixs[i,1]); append!(vals,1.0);

    #     # exp(k)*x_{2,i} - 2*h*g_{1,i}  = exp(k)*x{0,i}
    #     # x{0,i}*-exp(coef[ixs[1,i]])/h^2 = (x_{2,i} - 2*h*g_{1,i})*-exp(coef[ixs[1,i]])/h^2 



    #     append!(rows,ixs[i,1]);  append!(cols,ixs[i,1]);  append!(vals, 4*x[ixs[i,1]]*exp(coef[ixs[i,1]])/h^2); 
    #     append!(rows,ixs[i,1]);  append!(cols,ixs[i,1]);  append!(vals, -x[ixs[i,1+1]]*exp(coef[ixs[i,1]])/h^2); 
    #     append!(rows,ixs[i,1]);  append!(cols,ixs[i,1]);  append!(vals, -x[ixs[i,1+1]]*exp(coef[ixs[i,1]])/h^2); 
    #     append!(rows,ixs[i,1]);  append!(cols,ixs[i,1]);  append!(vals, -x[ixs[i-1,1]]*exp(coef[ixs[i,1]])/h^2); 
    #     append!(rows,ixs[i,1]);  append!(cols,ixs[i,1]);  append!(vals, -x[ixs[i+1,1]]*exp(coef[ixs[i,1]])/h^2); 

    #     # append!(rows,ixs[i,1]); append!(cols,ixs[i,1]); append!(vals,4*exp(coef[ixs[i,1]])/h^2);
    #     # append!(rows,ixs[i,1]); append!(cols,ixs[i,1+1]); append!(vals,-exp(coef[ixs[i,1]])/h^2);
    #     # append!(rows,ixs[i,1]); append!(cols,ixs[i,1+1]); append!(vals,-exp(coef[ixs[i,1]])/h^2);
    #     # append!(rows,ixs[i,1]); append!(cols,ixs[i-1,1]); append!(vals,-exp(coef[ixs[i,1]])/h^2);
    #     # append!(rows,ixs[i,1]); append!(cols,ixs[i+1,1]); append!(vals,-exp(coef[ixs[i,1]])/h^2);

    #     append!(rows,ixs[i,1]);  append!(cols,ixs[i+1,1]);  append!(vals, -exp(coef[ixs[i,1]])*(x[ixs[i+1,1]]/(4*h*h))); 
    #     append!(rows,ixs[i,1]);  append!(cols,ixs[i+1,1]);  append!(vals, +exp(coef[ixs[i,1]])*(x[ixs[i-1,1]]/(4*h*h))); 


    #     append!(rows,ixs[i,1]);  append!(cols,ixs[i-1,1]);  append!(vals, +exp(coef[ixs[i,1]])*(x[ixs[i+1,1]]/(4*h*h))); 
    #     append!(rows,ixs[i,1]);  append!(cols,ixs[i-1,1]);  append!(vals, -exp(coef[ixs[i,1]])*(x[ixs[i-1,1]]/(4*h*h))); 


    #     append!(rows,ixs[i,1]);  append!(cols,ixs[i,1]);  append!(vals, -exp(coef[ixs[i,1]])*(coef[ixs[i+1,1]]/(4h*h) - coef[ixs[i-1,1]]/(4h*h))*x[ixs[i+1,1]]); 
    #     append!(rows,ixs[i,1]);  append!(cols,ixs[i,1]);  append!(vals, +exp(coef[ixs[i,1]])*(coef[ixs[i+1,1]]/(4h*h) - coef[ixs[i-1,1]]/(4h*h))*x[ixs[i-1,1]]); 



    #     # append!(rows,ixs[i,1]); append!(cols,ixs[i+1,1]); append!(vals,-exp(coef[ixs[i,1]])*(coef[ixs[i+1,1]]/(4h*h) - coef[ixs[i-1,1]]/(4h*h)));
    #     # append!(rows,ixs[i,1]); append!(cols,ixs[i-1,1]); append!(vals,+exp(coef[ixs[i,1]])*(coef[ixs[i+1,1]]/(4h*h) - coef[ixs[i-1,1]]/(4h*h)));

        
      

    # end

    # #Vasen ylänurkka.
    # for i = 1:1

    #     append!(rows,ixs[i,1]);  append!(cols,ixs[i,1]);  append!(vals, 4*x[ixs[i,1]]*exp(coef[ixs[i,1]])/h^2); 
    #     append!(rows,ixs[i,1]);  append!(cols,ixs[i,1]);  append!(vals, -x[ixs[i,1+1]]*exp(coef[ixs[i,1]])/h^2); 
    #     append!(rows,ixs[i,1]);  append!(cols,ixs[i,1]);  append!(vals, -x[ixs[i,1+1]]*exp(coef[ixs[i,1]])/h^2); 
    #     append!(rows,ixs[i,1]);  append!(cols,ixs[i,1]);  append!(vals, -x[ixs[i+1,1]]*exp(coef[ixs[i,1]])/h^2); 
    #     append!(rows,ixs[i,1]);  append!(cols,ixs[i,1]);  append!(vals, -x[ixs[i+1,1]]*exp(coef[ixs[i,1]])/h^2); 


    #     # append!(rows,ixs[i,1]); append!(cols,ixs[i,1]); append!(vals,4*exp(coef[ixs[i,1]])/h^2);
    #     # append!(rows,ixs[i,1]); append!(cols,ixs[i,1+1]); append!(vals,-exp(coef[ixs[i,1]])/h^2);
    #     # append!(rows,ixs[i,1]); append!(cols,ixs[i,1+1]); append!(vals,-exp(coef[ixs[i,1]])/h^2);
    #     # append!(rows,ixs[i,1]); append!(cols,ixs[i+1,1]); append!(vals,-exp(coef[ixs[i,1]])/h^2);
    #     # append!(rows,ixs[i,1]); append!(cols,ixs[i+1,1]); append!(vals,-exp(coef[ixs[i,1]])/h^2);


    # end

    # #Yläreuna.
    # for i = 2:N-1

    #     append!(rows,ixs[1,i]);  append!(cols,ixs[1,i]);  append!(vals, 4*x[ixs[1,i]]*exp(coef[ixs[1,i]])/h^2); 
    #     append!(rows,ixs[1,i]);  append!(cols,ixs[1,i]);  append!(vals, -x[ixs[1+1,i]]*exp(coef[ixs[1,i]])/h^2); 
    #     append!(rows,ixs[1,i]);  append!(cols,ixs[1,i]);  append!(vals, -x[ixs[1+1,i]]*exp(coef[ixs[1,i]])/h^2); 
    #     append!(rows,ixs[1,i]);  append!(cols,ixs[1,i]);  append!(vals, -x[ixs[1,i+1]]*exp(coef[ixs[1,i]])/h^2); 
    #     append!(rows,ixs[1,i]);  append!(cols,ixs[1,i]);  append!(vals, -x[ixs[1,i-1]]*exp(coef[ixs[1,i]])/h^2); 

    #     # append!(rows,ixs[1,i]); append!(cols,ixs[1,i]); append!(vals,4*exp(coef[ixs[1,i]])/h^2);
    #     # append!(rows,ixs[1,i]); append!(cols,ixs[1,i+1]); append!(vals,-exp(coef[ixs[1,i]])/h^2);
    #     # append!(rows,ixs[1,i]); append!(cols,ixs[1,i-1]); append!(vals,-exp(coef[ixs[1,i]])/h^2);
    #     # append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,-exp(coef[ixs[1,i]])/h^2);
    #     # append!(rows,ixs[1,i]); append!(cols,ixs[1+1,i]); append!(vals,-exp(coef[ixs[1,i]])/h^2);

    #     append!(rows,ixs[1,i]);  append!(cols,ixs[1,i+1]);  append!(vals, -exp(coef[ixs[1,i]])*(x[ixs[1,i+1]]/(4*h*h))); 
    #     append!(rows,ixs[1,i]);  append!(cols,ixs[1,i+1]);  append!(vals, +exp(coef[ixs[1,i]])*(x[ixs[1,i-1]]/(4*h*h))); 


    #     append!(rows,ixs[1,i]);  append!(cols,ixs[1,i-1]);  append!(vals, +exp(coef[ixs[1,i]])*(x[ixs[1,i+1]]/(4*h*h))); 
    #     append!(rows,ixs[1,i]);  append!(cols,ixs[1,i-1]);  append!(vals, -exp(coef[ixs[1,i]])*(x[ixs[1,i-1]]/(4*h*h))); 


    #     append!(rows,ixs[1,i]);  append!(cols,ixs[1,i]);  append!(vals, -exp(coef[ixs[1,i]])*(coef[ixs[1,i+1]]/(4h*h) - coef[ixs[1,i-1]]/(4h*h))*x[ixs[1,i+1]]); 
    #     append!(rows,ixs[1,i]);  append!(cols,ixs[1,i]);  append!(vals, +exp(coef[ixs[1,i]])*(coef[ixs[1,i+1]]/(4h*h) - coef[ixs[1,i-1]]/(4h*h))*x[ixs[1,i-1]]); 



    #     # append!(rows,ixs[1,i]); append!(cols,ixs[1,i+1]); append!(vals,-exp(coef[ixs[1,i]])*(coef[ixs[1,i+1]]/(4h*h) - coef[ixs[1,i-1]]/(4h*h)));
    #     # append!(rows,ixs[1,i]); append!(cols,ixs[1,i-1]); append!(vals,+exp(coef[ixs[1,i]])*(coef[ixs[1,i+1]]/(4h*h) - coef[ixs[1,i-1]]/(4h*h)));

  
    # end

    # #Oikea reuna.
    # for i = 2:N-1
  
    #     append!(rows,ixs[i,N]);  append!(cols,ixs[i,N]);  append!(vals, 4*x[ixs[i,N]]*exp(coef[ixs[i,N]])/h^2); 
    #     append!(rows,ixs[i,N]);  append!(cols,ixs[i,N]);  append!(vals, -x[ixs[i-1,N]]*exp(coef[ixs[i,N]])/h^2); 
    #     append!(rows,ixs[i,N]);  append!(cols,ixs[i,N]);  append!(vals, -x[ixs[i+1,N]]*exp(coef[ixs[i,N]])/h^2); 
    #     append!(rows,ixs[i,N]);  append!(cols,ixs[i,N]);  append!(vals, -x[ixs[i,N-1]]*exp(coef[ixs[i,N]])/h^2); 
    #     append!(rows,ixs[i,N]);  append!(cols,ixs[i,N]);  append!(vals, -x[ixs[i,N-1]]*exp(coef[ixs[i,N]])/h^2); 

    #     # append!(rows,ixs[i,N]); append!(cols,ixs[i,N]); append!(vals,4*exp(coef[ixs[i,N]])/h^2);
    #     # append!(rows,ixs[i,N]); append!(cols,ixs[i,N-1]); append!(vals,-exp(coef[ixs[i,N]])/h^2);
    #     # append!(rows,ixs[i,N]); append!(cols,ixs[i,N-1]); append!(vals,-exp(coef[ixs[i,N]])/h^2);
    #     # append!(rows,ixs[i,N]); append!(cols,ixs[i+1,N]); append!(vals,-exp(coef[ixs[i,N]])/h^2);
    #     # append!(rows,ixs[i,N]); append!(cols,ixs[i-1,N]); append!(vals,-exp(coef[ixs[i,N]])/h^2);
        

    #     append!(rows,ixs[i,N]);  append!(cols,ixs[i+1,N]);  append!(vals, -exp(coef[ixs[i,N]])*(x[ixs[i+1,N]]/(4*h*h))); 
    #     append!(rows,ixs[i,N]);  append!(cols,ixs[i+1,N]);  append!(vals, +exp(coef[ixs[i,N]])*(x[ixs[i-1,N]]/(4*h*h))); 


    #     append!(rows,ixs[i,N]);  append!(cols,ixs[i-1,N]);  append!(vals, +exp(coef[ixs[i,N]])*(x[ixs[i+1,N]]/(4*h*h))); 
    #     append!(rows,ixs[i,N]);  append!(cols,ixs[i-1,N]);  append!(vals, -exp(coef[ixs[i,N]])*(x[ixs[i-1,N]]/(4*h*h))); 


    #     append!(rows,ixs[i,N]);  append!(cols,ixs[i,N]);  append!(vals, -exp(coef[ixs[i,N]])*(coef[ixs[i+1,N]]/(4h*h) - coef[ixs[i-1,N]]/(4h*h))*x[ixs[i+1,N]]); 
    #     append!(rows,ixs[i,N]);  append!(cols,ixs[i,N]);  append!(vals, +exp(coef[ixs[i,N]])*(coef[ixs[i+1,N]]/(4h*h) - coef[ixs[i-1,N]]/(4h*h))*x[ixs[i-1,N]]); 


    #     # append!(rows,ixs[i,N]); append!(cols,ixs[i+1,N]); append!(vals,-exp(coef[ixs[i,N]])*(coef[ixs[i+1,N]]/(4h*h) - coef[ixs[i-1,N]]/(4h*h)));
    #     # append!(rows,ixs[i,N]); append!(cols,ixs[i-1,N]); append!(vals,+exp(coef[ixs[i,N]])*(coef[ixs[i+1,N]]/(4h*h) - coef[ixs[i-1,N]]/(4h*h)));

    # end 

    # #Oikea ylänurkka.
    # for i = 1:1

  

    #     append!(rows,ixs[i,N]);  append!(cols,ixs[i,N]);  append!(vals, 4*x[ixs[i,N]]*exp(coef[ixs[i,N]])/h^2); 
    #     append!(rows,ixs[i,N]);  append!(cols,ixs[i,N]);  append!(vals, -x[ixs[i,N-1]]*exp(coef[ixs[i,N]])/h^2); 
    #     append!(rows,ixs[i,N]);  append!(cols,ixs[i,N]);  append!(vals, -x[ixs[i,N-1]]*exp(coef[ixs[i,N]])/h^2); 
    #     append!(rows,ixs[i,N]);  append!(cols,ixs[i,N]);  append!(vals, -x[ixs[i+1,N]]*exp(coef[ixs[i,N]])/h^2); 
    #     append!(rows,ixs[i,N]);  append!(cols,ixs[i,N]);  append!(vals, -x[ixs[i+1,N]]*exp(coef[ixs[i,N]])/h^2); 


    #     # append!(rows,ixs[i,N]); append!(cols,ixs[i,N]); append!(vals,4*exp(coef[ixs[i,N]])/h^2);
    #     # append!(rows,ixs[i,N]); append!(cols,ixs[i,N-1]); append!(vals,-exp(coef[ixs[i,N]])/h^2);
    #     # append!(rows,ixs[i,N]); append!(cols,ixs[i,N-1]); append!(vals,-exp(coef[ixs[i,N]])/h^2);
    #     # append!(rows,ixs[i,N]); append!(cols,ixs[i+1,N]); append!(vals,-exp(coef[ixs[i,N]])/h^2);
    #     # append!(rows,ixs[i,N]); append!(cols,ixs[i+1,N]); append!(vals,-exp(coef[ixs[i,N]])/h^2);
        
    # end
    
    # #Alareuna.
    # # for i = 1:N
  
    # #     # M[ixs[N,i],ixs[N,i]] = 1
    # #     #Boff[ixs[N,i],ixs[N,i]] = 0
    # #     #B[ixs[N,i],ixs[N,i]] = 1.0
    # #     #y[ixs[N,i]] = 100

    # #     #append!(rows,ixs[N,i]); append!(cols,ixs[N,i]); append!(vals,1.0);

    # # end
    
    
    return sparse(rows,cols,vals,N^2,N^2)



    #return M
end


function adgradi(X,Y,co,rhs,meas,F,icova)

    #M,y = feikki(co,f)
    M,y = solution(X,Y,co,rhs)
    x = M\y

    dEdx = M# equderix(co,f,x)
    dEdp = equderipar(X,Y,co,rhs,x)
  
    dFdx = F'*icova*(meas-F*x)

    lambda = (dEdx')\(dFdx)

    return -lambda'*dEdp

end 

function adgradiexp(X,Y,colog,rhs,meas,F,icova)

    #M,y = feikki(co,f)
    M,y = solutionexp(X,Y,colog,rhs)
    sol = M\y

    dEdx = M# equderix(co,f,x)
    dEdp = equderiparexp(X,Y,colog,rhs,sol)
  
    dFdx = F'*icova*(meas-F*sol)

    lambda = (dEdx')\(dFdx)


    return -lambda'*dEdp

end

function loglike(co,rhs,meas,F,icova)

    M,y = solution(co,rhs)
    x = M\y   
    xs  = F*x

    return -1/2*(meas-xs)'*icova*(meas-xs)

end 

function loglikeexp(X,Y,colog,rhs,meas,F,icova)

    M,y = solutionexp(X,Y,colog,rhs)
    #M,y = feikki(co,f)
    sol = M\y   
    xs  = F*sol

    return -1/2*(meas-xs)'*icova*(meas-xs)

end

function logpisodiff(lf,ulf,x,args,cache)
    noisesigma = args.noisesigma
    #scale = args.scale; 
    y  = args.y; F = args.F
    #bscale = args.bscale
    Dx = args.Dx; Dy = args.Dy
    Db = args.Db
    X = args.X; Y = args.Y
    rhs = args.rhs;

    res = cache.residual
    Dxprop = cache.Dxprop
    Dyprop = cache.Dyprop
    Dbprop = cache.Dbprop
    Fxprop = cache.Fxprop
    ld = cache.ld
    lb = cache.lb


    ####M,r = solutionexp(X,Y,x,rhs)
    M,r = solution(X,Y,x,rhs)
    pdesol = M\r 

    mul!(Fxprop,F,pdesol)
    #res .= Fxprop - y  
    storesubst!(res,Fxprop,y)
    #Dxprop .= Dx*x
    mul!(Dxprop,Dx,x)
    #Dyprop .= Dy*x
    mul!(Dyprop,Dy,x)
    #Dbprop .= Db*x
    mul!(Dbprop,Db,x)

    Nd = length(Dyprop)
    #=Threads.@threads=# for i = 1:Nd
        ld[i] =  lf(etaisyys(Dxprop[i],Dyprop[i]))
    end

    Nb = length(Dbprop)
    #=Threads.@threads=# for i = 1:Nb
        lb[i] =  ulf(Dbprop[i])
    end

    return   -0.5/noisesigma^2*dotitself(res)  + sum(ld)  + sum(lb)  

end

function  logpisodiffgradi(lf,ulf,glf,gulf,x,args,cache;both=true)
    #@info minimum(x), maximum(x)
    noisesigma = args.noisesigma
    y  = args.y; F = args.F
    Dx = args.Dx; Dy = args.Dy
    Db = args.Db
    X = args.X; Y = args.Y
    rhs = args.rhs; icova = I(length(y))*1/noisesigma^2

    logp = 0.0

    res = cache.residual
    Dxprop = cache.Dxprop
    Dyprop = cache.Dyprop
    Dbprop = cache.Dbprop
    Fxprop = cache.Fxprop
    G = cache.gradiprop
    ld = cache.ld
    lb = cache.lb
    gld = cache.gld
    glb = cache.glb

    ###M,r = solutionexp(X,Y,x,rhs)
    M,r = solution(X,Y,x,rhs)

    pdesol = M\r 

    dEdx = M
    ###dEdp = equderiparexp(X,Y,x,rhs,pdesol)
    dEdp = equderipar(X,Y,x,rhs,pdesol)
  
    dFdx = F'*icova*(y-F*pdesol)

    lambda = (dEdx')\(dFdx)
    G .= (-lambda'*dEdp)[:]

    #Fxprop .= F*x
    mul!(Fxprop,F,pdesol)
    storesubst!(res,Fxprop,y)
    #res .= Fxprop - y  
    #Dxprop .= Dx*x
    mul!(Dxprop,Dx,x)
    #Dyprop .= Dy*x
    mul!(Dyprop,Dy,x)
    #Dbprop .= Db*x
    mul!(Dbprop,Db,x)

    #G .= F'*(-((res)./noisesigma.^2))
    #mul!(G,F',-((res)/noisesigma.^2))

    Nd = length(Dyprop)
    #=Threads.@threads=# for i = 1:Nd
        p = etaisyys(Dxprop[i],Dyprop[i])
        ld[i] =  lf(p)
        gld[i] = p == 0.0 ? 0.0 : glf(p)/p # Nollalla jako!?
    end

    Nb = length(Dbprop)
    #=Threads.@threads=# for i = 1:Nb
        r = Dbprop[i]
        lb[i] =  ulf(r)
        glb[i] = gulf(r)
    end

    if both
        logp = -0.5/noisesigma^2*dotitself(res) + sum(ld)  + sum(lb)  
    end

    mul!(G,Db',glb,1,1)
    mul!(G,Dx',Dxprop.*gld,1,1)
    mul!(G,Dy',Dyprop.*gld,1,1)
    #println(maximum(gld),",",minimum(gld))

    return logp, G 
end

function upbarrier(x)
    r = 0.0
    x <= 0 ? r = Inf : r= -log(x)
    return r
end

function gupbarrier(x)
    g = 0.0
    x <= 0 ? g = -Inf : g = 1/x
    return g
end

mu = 1e-5




linspace(x,y,z) = range(x,stop=y,length=z)

Random.seed!(1)

N = 128
Nbig = 223
#Nmeas = 75

Ntarg = 25


x = range(0,stop=6,length=N)
Y,X = meshgrid(x,x)
Y = reverse(Y,dims=1)
rY = reverse(Y)


xbig = range(0,stop=6,length=Nbig)
Ybig,Xbig = meshgrid(xbig,xbig)
Ybig = reverse(Ybig,dims=1)


cw = 150.0
noisevar = 1e-3

#H=ones(size(X))
#H[8:8:64-8, 8:8:64-8] .= 1
#H[Int(N/8):Int(N/8):Int(N-N/8),Int(N/8):Int(N/8):Int(N-N/8)] .= 1
# ixs = findall(H .== 1)
# Xt = X[ixs]
# Yt = Y[ixs]

kernel(xi,xj,yi,yj;constant=cw) = constant/pi*exp(-constant*((xi-xj)^2 + (yi-yj)^2) )
#tf(x,y) = @. Float64(( sqrt((x-3)^2 + (y-2.5)^2) < 2 )*(2) + 1)
tf(x,y) = @. log(1 +  0.0*Float64(sqrt( (x-3)^2 + (y-3.5)^2  ) < 1.5) + 0.0*Float64(sqrt( (x-4)^2 + (y-0.5)^2  ) < 1.0) ) #-0.5 + sin(2*x)*cos(y)
#rhsfun(x,y) = @. (4 ≤ x ≤ 5)*137 + (5 <  y ≤ 6)*274
#rhsfun(x,y) = @. (4 ≤ y ≤ 5)*10 + (5 <  y ≤ 6)*27
rhsfun(x,y) = @. 10*exp(- ((x-2)^2 + (y-3.3)^2 ) ) + 10*exp(- ((x-4)^2 + (y-4.3)^2 ) ) + 10*exp(-0.5* ((x-3)^2 + (y-0.5)^2 ) )
#rhsfun(x,y) = @. 5+ 10*cos(x) 

RHS = rhsfun(X,Y)
rhs = RHS[:]

RHSbig = rhsfun(Xbig,Ybig)
rhsbig = RHSbig[:]

cologbig = tf(Xbig,Ybig)
#colog = tf(X,Y)
#co = exp.(colog)[:]

MC  = spdematrix(xbig,0.8,1);
kp = 0.2*abs.(MC\randn(Nbig*Nbig)*100 .+ 1);
gpart = @. sqrt( (Xbig-3)^2 + (Ybig-3.5)^2  ) < 1.7 || (sqrt( (Xbig-3)^2 + (Ybig-2)^2  )) < 1.3
cologbig[gpart] = cologbig[gpart] + reshape(kp,Nbig,Nbig)[gpart]



Mbig,rbig = solutionexp(Xbig,Ybig,cologbig[:],rhsbig)
solbig = Mbig\rbig
solbig = reshape(solbig,(Nbig,Nbig))

interp_f = LinearInterpolation((xbig,xbig), cologbig)
interp_f2 = CubicSplineInterpolation((xbig,xbig), cologbig)
colog = interp_f(x,x)

M,r = solutionexp(X,Y,colog[:],rhs)
sol = M\r
K = reshape(sol,(N,N))

interp_meas = LinearInterpolation((xbig,xbig), solbig)

# xt = range(0.5,stop=5.5,length=Nmeas)
# Yt,Xt = meshgrid(xt,xt)
# Yt = reverse(Yt,dims=1)

# Nth = trunc(Int64,(N-2)/Ntarg)
# xt = x[2:Nth:end-1]
# Yt,Xt = meshgrid(xt,xt)
# Yt = reverse(Yt,dims=1)
# Aix = LinearIndices((1:N,1:N))
# Ix = Aix[2:Nth:end-1,2:Nth:end-1]

# is = interp_meas(xt,xt)



using QuasiMonteCarlo
axt = x[2:end-1]
tps = QuasiMonteCarlo.sample(Ntarg^2,[axt[1],axt[1]],[axt[end],axt[end]],LatticeRuleSample())
Pv = Vector{Vector{Float64}}(undef,N*N)
for i in eachindex(X)
    Pv[i] = zeros(2)
    Pv[i][1] = rY[i]
    Pv[i][2] = X[i]
end
Pix = mapslices(q-> findmin(t->norm(t-q),Pv)[2],tps,dims=1)
Pix = collect(Set(Pix))[:]
sort!(Pix)
P = [[rY[i],X[i]] for i in Pix]

Nallmeas = length(Pix)
yt = map(t-> t[1],P)
xt = map(t-> t[2],P)
is = zeros(Nallmeas)
for i in eachindex(P)
    is[i] = interp_meas(P[i][1],P[i][2])
end
Ix = Pix


Nallmeas = length(Ix)

F = sparse(Vector(1:Nallmeas), Ix[:], ones(Nallmeas), Nallmeas,N*N)
y = is[:] + randn(Nallmeas,)*sqrt(noisevar)

#imshow(reshape(F*sol-is[:],Int64(sqrt(length(is))),Int64(sqrt(length(is)))))

stop()

tall0 = "pdet/muut.mat"
matwrite(tall0,Dict("y"=>y,"solbig"=>solbig,"rhsbig"=>rhsbig,"cologbig"=>cologbig,"xt"=>xt,"yt"=>yt))


x0 =  median(exp.(cologbig))*ones(N*N)

# coef = @. exp(-0.5 + sin(2*X)*cos(Y))
# co = coef[:]
# colog = @. log(co)


cova = I(size(F)[1])*noisevar
icova = inv(cova)
# meas = F*sol + cholesky(cova).L*randn(size(A)[1])
fun(co) = loglike(co,rhs,y,F,icova)
funexp(colog) = loglikeexp(X,Y,colog,rhs,y,F,icova)



include("2dapprox.jl")
include("1dapprox.jl")

function interp_help1d()
    if !@isdefined ip1d
        
        return polator1d(transition=true)
  
    else
        return ip1d
    end

end


function interp_help()
    if !@isdefined ip
        
        return polator(transition=true)
  
    else
        return ip
    end

end


const ip1d = interp_help1d()
const ip = interp_help()





macro bscale()
    return :(1.0)
end




for scale ∈  [ 0.01, 0.05, 0.1, 0.5]
    for stab ∈ [0.51, 0.8, 1.0, 1.4, 1.7 , 1.9]
        
        # scale = 0.003
        # stab = 0.7
      
        ala = 0.0001*ones(N*N)
        yla = 100*ones(N*N)

        Dx,Dy,Db = regmatrices_first(N);
        arg = (X=X,Y=Y,rhs=rhs,F=F,Db=Db,Dx=Dx,Dy=Dy,scale=scale,bscale=@bscale(),y=y,noisesigma=sqrt(noisevar))
        cac = (Dbprop=Db*x0,Dyprop=Dy*x0,Dxprop=Dx*x0,Fxprop=F*x0,ld=similar(Dx*x0),lb=similar(Db*x0),gld=similar(Dx*x0), glb=similar(Db*x0), gradiprop = similar(x0), residual=similar(y))
        cac2 = deepcopy(cac)

        lf(x) = logdensity(stab,abs(x),ip;gamma0=scale)
        ulf(x) = logdensityuni(stab,abs(x),ip1d;gamma0=@bscale())
        glf(x) = logdensityderi(stab,x,ip;gamma0=scale)
        gulf(x) = logdensityunideri(stab,x,ip1d;gamma0=@bscale())


        target(x) = -logpisodiff(lf,ulf,x,arg,cac)
        targetgrad(x) = -logpisodiffgradi( lf,ulf,glf,gulf,x,arg,cac2)[2]

        
       

        ### LBFGSB -->

        function gr!(z,x)
            G = -logpisodiffgradi( lf,ulf,glf,gulf,x,arg,cac2)[2]
            N = length(x)
            @inbounds @simd for i = 1:N
                z[i] = G[i]
            end
        end
        
        function ff(x)
            return -logpisodiff(lf,ulf,x,arg,cac)
        end
        
        n = N*N
        optimizer = L_BFGS_B(n, 17)

        x00 = copy(x0)  # the initial guess
        # set up bounds
        bounds = zeros(3, n)
        for i = 1:n
            bounds[1,i] = 2.0  # represents the type of bounds imposed on the variables: #  0->unbounded, 1->only lower bound, 2-> both lower and upper bounds, 3->only upper bound
            bounds[2,i] = ala[i]
            bounds[3,i] = yla[i]
        end

        global minf, minx = optimizer(ff, gr!, x00, bounds, m=17, factr=1e7, pgtol=1e-2, iprint=1, maxfun=50000, maxiter=2000)


        tall = "pdet/"*string(stab)*"-"*string(scale)*"-"*string(N)*"-lbfgs.mat"
        matwrite(tall,Dict("MAP"=>minx))


 end
end

