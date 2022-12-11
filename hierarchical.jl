using Random
using QuadGK
using PyPlot
using Statistics
using Interpolations
using LinearAlgebra
using MAT
using ProgressBars
using Optim
using SparseArrays
using SpecialFunctions
using AlphaStableDistributions
using UnPack

cd( @__DIR__)

function cb(iter,gt,data)
    println(iter.iteration, ":\t ", iter.value, ",\t ", iter.g_norm)
    x = iter.metadata["x"]
    sumsq = norm(x[1:N]-gt)#/norm(gt) #sum((x.-gt).^2)
    append!(data["resi"],sumsq)
    append!(data["gnorm"],iter.g_norm)
    append!(data["logd"],iter.value)

    return false
end

function difference1(X)
    N = length(X);
    M = zeros(N,N);

    for i  = 2:N
       M[i,i] = -1;
       M[i,i-1] = 1;
    end
    M[1,1] = 1;
    return M
end

@inline @fastmath function sigmoid(x)
    if x < 0.0
        a = exp(x) 
        return a / (1.0 + a) 
    else 
        return 1.0 / (1.0 + exp(-x))
    end
    
end

@inline @fastmath function dsigmoid(x)
    q = sigmoid(x)
    return q*(1-q)
end

function measurementmatrix(X,MX,kernel)
    s = X[2]-X[1];
    F = zeros(size(MX)[1],size(X)[1])
    for i = 1:size(MX)[1]
        F[i,:] = s.*kernel.(X,MX[i]);
    end
    return F
end

# function regmatrices_first(dim)
#     reg1d = spdiagm(Pair(0,-1*ones(dim))) + spdiagm(Pair(1,ones(dim-1))) + spdiagm(Pair(-dim+1,ones(1))) ;reg1d[dim,dim] = 0
#     #reg1d = reg1d[1:dim-1,:]
#     iden = I(dim)
#     regx = kron(reg1d,iden)
#     regy = kron(iden,reg1d)

#     rmxix = sum(abs.(regx) ,dims=2) .< 2
#     rmyix = sum(abs.(regy) ,dims=2) .< 2
#     boundary = ((rmxix + rmyix)[:]) .!= 0
#     q = findall(boundary .== 1)
#     regx = regx[setdiff(1:dim^2,q), :] 
#     regy = regy[setdiff(1:dim^2,q), :] 
    
#     s = length(q)
#     bmatrix = sparse(zeros(s,dim*dim))
#     for i=1:s
#         v = q[i]
#         bmatrix[i,v] = 1
#     end
#     #bmatrix = bmatrix[i,i] .= 1

#     return regx,regy,bmatrix
# end

@inbounds @inline function storevector!(target,source)
    N = length(target)
    @assert N == length(source)
    @simd for i = 1:N
        target[i] = source[i]
    end
end

@inbounds @inline function storesubst!(target,a1,a2)
    N = length(target)
    @assert N == length(a1) == length(a2)
    @simd for i = 1:N
        target[i] = a1[i] - a2[i]
    end
end

@inbounds @inline function storeadd!(target,a1,a2)
    N = length(target)
    @assert N == length(a1) == length(a2)
    @simd for i = 1:N
        target[i] = a1[i] + a2[i]
    end
end


@inbounds @inline function slicemadd!(target,cols,vec,ix)
    N = length(ix)
    @assert N == length(vec)
    #mul!(target,cols[ix[1]],vec[1])
    for i = 1:N
        mul!(target,cols[ix[i]],vec[i],1,1)
    end
end

@inline function dotitself(x::Array{Float64,1})  
    return dot(x,x)
end



@inline  function etaisyys(x,y)
    return sqrt(x^2 + y^2)
end





function deriv(f,x0::T;eps=1e-9) where T
    eps = convert(T,eps)
    d = (f(x0+eps)-f(x0-eps))/(2*eps)
    return d
end

function derivb(f,x0::T;eps=1e-9) where T
    eps = convert(T,eps)
    d = (f(x0)-f(x0-eps))/(eps)
    return d
end



function logdensityunideriscale(alpha,r0,ip;gamma0=1.0)
    interp = ip.uinterp; interp2 = ip.uinterp2; interp3 = ip.uinterp3;  interp4 = ip.uinterp4; interp5 = ip.uinterp5; interp6 = ip.uinterp6
    bou0 = ip.bou0
    bou1 = ip.bou1
    bou2 = ip.bou2

    r = abs(r0/gamma0)
    R = r

    if alpha == 2.0
        return  -1/gamma0  + (r*(r + abs(r0)/gamma0)/(2*gamma0^3)) 
    elseif  (bou1>=r>=bou0)   && (alpha <= 1.995)
        return -abs(r0)/gamma0^2*(Interpolations.gradient(interp,alpha,r))[2] -1/(gamma0)
    #elseif  (bou1>=r>=bou0)  && (alpha > 1.995)
    #    return -abs(r0)/gamma0^2*(Interpolations.gradient(interp2,alpha,r))[2] -1/(gamma0)
    elseif (0<=r<bou0)  && (alpha <= 1.995)
        return -abs(r0)/gamma0^2*(Interpolations.gradient(interp3,alpha,r))[2] -1/(gamma0)
    #elseif (0<=r<bou0)  && (alpha > 1.995)
     #   return -abs(r0)/gamma0^2*(Interpolations.gradient(interp6,alpha,r))[2] -1/(gamma0)
    elseif (bou2>=r>=bou1)  && (alpha <= 1.995)
        return -abs(r0)/gamma0^2*(Interpolations.gradient(interp4,alpha,r))[2] -1/(gamma0)
    #elseif (bou2>=r>=bou1)  && (alpha > 1.995)
    #    return -abs(r0)/gamma0^2*(Interpolations.gradient(interp5,alpha,r))[2] -1/(gamma0)
    elseif (r>bou2) && (alpha <= 1.995)
        r = Inf
        logr = log(R)
        #c = -(log(pi) + logr)

        l0 =  loggamma(alpha+1) + log(sin(pi*(alpha)/2)) -(alpha)*log(R)
        l0g = -(alpha+1)*1/R
        li = 0.0
        lig = 0.0
        for n = 2:4
            lcur = -loggamma(n+1) -logr*(alpha*n)  +loggamma(alpha*n+1)
            termcur = sin(n*pi*(alpha)/2)*(-1)^(n-1)
            stermcur = sign(termcur)
            lcur = lcur + log(stermcur*termcur)
            q = stermcur*exp(lcur-l0)
            li = li + q
            lig = lig + q*(-l0g + -1/R*(alpha*n+1))
        end
        return  (l0g + lig/(1+li))*(-abs(r0)/(gamma0^2))  -1/(gamma0)

        #*(-R/(gamma0^2))

        # #l0 = loggamma(alpha+1) + log(-cos(pi*(alpha+1)/2)) -(alpha+1)*log(r)
        # l0 =  loggamma(alpha+1) + log(sin(pi*(alpha)/2)) -(alpha)*log(r)
        # l0g = -(alpha)*1/r
        # logr = log(r)
        # li = 0.0
        # lig = 0.0
        # for n = 2:4
        #     #lcur = -logfactorial(n) -log(r)*(alpha*n+1) + log(abs(cos(pi*(alpha*n+1)/2))) +loggamma(alpha*n+1)
        #     lcur = -loggamma(n+1) -logr*(alpha*n)  +loggamma(alpha*n+1)
        #     termcur = sin(n*pi*(alpha)/2)*(-1)^(n-1)
        #     stermcur = sign(termcur)
        #     lcur = lcur + log(stermcur*termcur)
        #     q = stermcur*exp(lcur-l0)
        #     #q = exp(lcur-l0)
        #     li = li + q
        #     lig = lig + q*(-l0g + -1/r*(alpha*n))
        #     #lig = lig + q*(-l0g + -1/r*(alpha*n+1))
        # end
        # return -(l0g + lig/(1+li))*abs(r0)/gamma0^2 -1/(gamma0)
    else 
        error("Alpha not in allowed range")
    end
end

function logdensityunideristabi(alpha,r0,ip;gamma0=1.0)
    interp = ip.uinterp; interp2 = ip.uinterp2; interp3 = ip.uinterp3;  interp4 = ip.uinterp4; interp5 = ip.uinterp5; interp6 = ip.uinterp6
    bou0 = ip.bou0
    bou1 = ip.bou1
    bou2 = ip.bou2

    r = abs(r0/gamma0)
    if (bou1>=r>=bou0)  && (alpha <= 1.995)
        return Interpolations.gradient(interp,alpha,r)[1]
    #elseif (bou1>=r>=bou0)  && (alpha > 1.995)
    #    return Interpolations.gradient(interp2,alpha,r)[1]
    elseif (0<=r<bou0)  && (alpha <= 1.995)
        return Interpolations.gradient(interp3,alpha,r)[1]
    #elseif (0<=r<bou0)  && (alpha > 1.995)
    #    return Interpolations.gradient(interp6,alpha,r)[1]
    elseif (bou2>=r>=bou1)  && (alpha <= 1.995)
        return Interpolations.gradient(interp4,alpha,r)[1]
    #elseif (bou2>=r>=bou1)  && (alpha > 1.995)
    #    return Interpolations.gradient(interp5,alpha,r)[1]
    elseif (r>bou2) && (alpha <= 1.995)
        logr = log(r)
        #l0 =  loggamma(alpha+1) + log(-cos(pi*(alpha+1)/2)) -(alpha+1)*logr
        l0 =  loggamma(alpha+1) + log(sin(pi*(alpha)/2)) -(alpha)*logr
        gl0 =  digamma(alpha+1) + 1/2*pi*cot(pi*alpha/2) -logr 
        li = 0.0 
        gli = 0.0
        for n = 2:4
            # lcur = -logfactorial(n) -logr*(alpha*n+1) + log(abs(cos(pi*(alpha*n+1)/2))) +loggamma(alpha*n+1)
            # glcur =  -logr*n + 1/2*pi*n*cot(pi*alpha*n/2) +n*digamma(n*alpha+1)
            # q = exp(lcur-l0)
            # li = li + q
            lcur = -loggamma(n+1) -logr*(alpha*n)  +loggamma(alpha*n+1)
            termcur = sin(n*pi*(alpha)/2)*(-1)^(n-1)
            stermcur = sign(termcur)
            lcur = lcur + log(stermcur*termcur)
            q = stermcur*exp(lcur-l0)
            li = li + q   
            glcur =  -logr*n + 1/2*pi*n*cot(pi*alpha*n/2) +n*digamma(n*alpha+1)
            gli = gli + (glcur - gl0)*q
        end
        #println(li)
        return gl0 + gli/(li+1)
        #return l0 + log1p(li) -log(gamma0) - 1.1447298858494002 #log(pi)
    else 
        error("Alpha not in allowed range")
    end
end




function gradi(f,x0;eps=1e-8)
    xp = copy(x0)
    f0 = f(x0)
    Nc = length(f0)
    Nv = length(x0)
    if ((Nc > 1) || ((ndims(x0) > 1)))
        J = zeros(Nc,Nv)
    else
        J = zeros(Nv,)
    end
    for i = 1:Nv
        orig = xp[i]
        xp[i] += eps
        fp = f(xp)
        if ((Nc > 1) || ((ndims(x0) > 1)))
            J[:,i] .= (fp-f0)./eps
        else
            J[i] = (fp-f0)/eps
        end
        xp[i] = orig
    end

    return J
end





function  logpdiffscale(funks,U,arguments,cache)
    Fprop = cache.Fprop
    Dprop = cache.Dprop
    Sprop = cache.Sprop
    Sfprop = cache.Sfprop
    res = cache.residual
    ld = cache.ld
    lc = cache.lc
    Nd = length(Dprop)
    x = view(U,1:Nd)
    s = view(U,Nd+1:2*Nd)
    @unpack condiscalelf, scalelf, sctf  =  funks

    M = arguments.M;  S = arguments.S; 
    noisesigma = arguments.noisesigma
    y  = arguments.y; F = arguments.F
    
    
    mul!(Fprop,F,x)
    mul!(Dprop,M,x)
    mul!(Sprop,S,s)
    storesubst!(res,Fprop,y)

    for i = 1:Nd
        Sfprop[i] =  sctf(s[i])
    end
    
    #=Threads.@threads=# for i = 1:Nd
        ld[i] =  condiscalelf(Dprop[i],Sfprop[i]) #sf -> s
        lc[i] =  scalelf(Sprop[i])
    end

    logp =  -0.5/noisesigma^2*dotitself(res)
    logp = logp + sum(ld) + sum(lc);

    return logp

end


function  logpdiffgradiscale(funks,U,arguments,cache;both=false)
    Fprop = cache.Fprop
    Dprop = cache.Dprop
    Sprop = cache.Sprop
    Sfprop = cache.Sfprop
    res = cache.residual
    @unpack gld, glc, ld, lc, gc = cache
    Nd = length(Dprop)
    G = cache.gradiprop
    Gx = view(G,1:Nd)
    Sx = view(G,Nd+1:2*Nd)
    x = view(U,1:Nd)
    s = view(U,Nd+1:2*Nd)
    logp = 0.0
    @unpack  gcondiscalelf, condiscalelf, scalelf,gscalelf, scalederi, sctf, gsctf  = funks

    M = arguments.M;  S = arguments.S; 
    noisesigma = arguments.noisesigma
    y  = arguments.y; F = arguments.F
       
    mul!(Fprop,F,x)
    mul!(Dprop,M,x)
    mul!(Sprop,S,s)
    storesubst!(res,Fprop,y)   

    mul!(Gx,F',-((res)/noisesigma.^2))

    for i = 1:Nd
        Sfprop[i] =  sctf(s[i])
    end

   
    #=Threads.@threads=# for i = 1:Nd
        ld[i] =  condiscalelf(Dprop[i],Sfprop[i])
        
        gld[i] =  gcondiscalelf(Dprop[i],Sfprop[i])
        gc[i] = gsctf(s[i])*scalederi(Dprop[i],Sfprop[i])

        lc[i] =  scalelf(Sprop[i])
        glc[i] = gscalelf(Sprop[i])
    end

    if both
        logp = -0.5/noisesigma^2*dotitself(res) + sum(ld) + sum(ls)
    end

    mul!(Gx,M',gld,1,1)
    mul!(Sx,S',glc)
    storeadd!(Sx,Sx,gc)

    return logp,G

end


function  logpdiffstabi(funks,U,arguments,cache)
    Fprop = cache.Fprop
    Dprop = cache.Dprop
    Sprop = cache.Sprop
    Sfprop = cache.Sfprop
    res = cache.residual
    ld = cache.ld
    ls = cache.ls
    Nd = length(Dprop)
    x = view(U,1:Nd)
    s = view(U,Nd+1:2*Nd)
    @unpack stabilf, condistabilf, sbtf  =  funks

    M = arguments.M;  S = arguments.S; 
    noisesigma = arguments.noisesigma
    y  = arguments.y; F = arguments.F
    
    
    mul!(Fprop,F,x)
    mul!(Dprop,M,x)
    mul!(Sprop,S,s)
    storesubst!(res,Fprop,y)

    for i = 1:Nd
        Sfprop[i] =  sbtf(s[i])
    end
    
    #=Threads.@threads=# for i = 1:Nd
        ld[i] =  condistabilf(Dprop[i],Sfprop[i]) 
        ls[i] =  stabilf(Sprop[i])
    end

    logp =  -0.5/noisesigma^2*dotitself(res)
    logp = logp + sum(ld) + sum(ls);

    return logp

end


function  logpdiffgradistabi(funks,U,arguments,cache;both=false)
    Fprop = cache.Fprop
    Dprop = cache.Dprop
    Sprop = cache.Sprop
    Sfprop = cache.Sfprop
    res = cache.residual
    @unpack gld, gls, ld, ls, gs = cache
    Nd = length(Dprop)
    G = cache.gradiprop
    Gx = view(G,1:Nd)
    Sx = view(G,Nd+1:2*Nd)
    x = view(U,1:Nd)
    s = view(U,Nd+1:2*Nd)
    logp = 0.0
    @unpack  stabideri, condistabilf, gcondistabilf, sbtf, gsbtf, stabilf, gstabilf  = funks

    M = arguments.M;  S = arguments.S; 
    noisesigma = arguments.noisesigma
    y  = arguments.y; F = arguments.F
       
    mul!(Fprop,F,x)
    mul!(Dprop,M,x)
    mul!(Sprop,S,s)
    storesubst!(res,Fprop,y)   

    mul!(Gx,F',-((res)/noisesigma.^2))

    for i = 1:Nd
        Sfprop[i] =  sbtf(s[i])
    end

   
    #=Threads.@threads=# for i = 1:Nd
        ld[i] =  condistabilf(Dprop[i],Sfprop[i])
        
        gld[i] = gcondistabilf(Dprop[i],Sfprop[i])
        gs[i] = gsbtf(s[i])*stabideri(Dprop[i],Sfprop[i])

        ls[i] =  stabilf(Sprop[i])
        gls[i] = gstabilf(Sprop[i])
    end

    if both
        logp = -0.5/noisesigma^2*dotitself(res) + sum(ld) + sum(ls)
    end

    mul!(Gx,M',gld,1,1)
    mul!(Sx,S',gls)
    storeadd!(Sx,Sx,gs)

    return logp,G

end

function  logpdiffboth(funks,U,arguments,cache)
    Fprop = cache.Fprop
    Dprop = cache.Dprop
    Sprop = cache.Sprop
    Sfprop = cache.Sfprop
    Cfprop = cache.Cfprop
    Cprop = cache.Cprop
    res = cache.residual
    @unpack gld, gls, ld, ls, lc = cache
    Nd = length(Dprop)
    x = view(U,1:Nd)
    s = view(U,Nd+1:2*Nd)
    c = view(U,2*Nd+1:3*Nd)
    @unpack stabilf,stabilf, scalelf, sbtf, sctf, condilf =  funks

    M = arguments.M;  S = arguments.S;  C = arguments.C
    noisesigma = arguments.noisesigma
    y  = arguments.y; F = arguments.F
    
    
    mul!(Fprop,F,x)
    mul!(Dprop,M,x)
    mul!(Sprop,S,s)
    mul!(Cprop,C,c)
    storesubst!(res,Fprop,y)

    for i = 1:Nd
        Sfprop[i] =  sbtf(s[i])
        Cfprop[i] =  sctf(c[i])
    end
    
    #=Threads.@threads=# for i = 1:Nd
        ld[i] =  condilf(Dprop[i],Sfprop[i],Cfprop[i]) 
        ls[i] =  stabilf(Sprop[i])
        lc[i] =  scalelf(Cprop[i])
    end

    logp =  -0.5/noisesigma^2*dotitself(res)
    logp = logp + sum(ld) + sum(ls) + sum(lc);

    return logp

end

function  logpdiffgradiboth(funks,U,arguments,cache;both=false)
    Fprop = cache.Fprop
    Dprop = cache.Dprop
    Sprop = cache.Sprop
    Sfprop = cache.Sfprop
    Cfprop = cache.Cfprop
    Cprop = cache.Cprop
    res = cache.residual
    @unpack gld, gls, glc, lc, ld, ls, gc, gs = cache
    Nd = length(Dprop)
    G = cache.gradiprop
    Gx = view(G,1:Nd)
    Sx = view(G,Nd+1:2*Nd)
    Cx = view(G,2*Nd+1:3*Nd)
    x = view(U,1:Nd)
    s = view(U,Nd+1:2*Nd)
    c = view(U,2*Nd+1:3*Nd)
    logp = 0.0
    @unpack  condilf, stabideri_both, scalederi_both, gcondilf, sbtf, gsbtf, sctf, gsctf, stabilf, gstabilf, scalelf, gscalelf  = funks

    M = arguments.M;  S = arguments.S;  C = arguments. C
    noisesigma = arguments.noisesigma
    y  = arguments.y; F = arguments.F
     
    mul!(Fprop,F,x)
    mul!(Dprop,M,x)
    mul!(Sprop,S,s)
    mul!(Cprop,C,c)
    storesubst!(res,Fprop,y)

    mul!(Gx,F',-((res)/noisesigma.^2))

    for i = 1:Nd
        Sfprop[i] =  sbtf(s[i])
        Cfprop[i] =  sctf(c[i])
    end

   
    #=Threads.@threads=# for i = 1:Nd
        ld[i] =  condilf(Dprop[i],Sfprop[i],Cfprop[i])     
        gld[i] = gcondilf(Dprop[i],Sfprop[i],Cfprop[i])

        lc[i] =  scalelf(Cprop[i])
        gc[i] = gsctf(c[i])*scalederi_both(Dprop[i],Sfprop[i],Cfprop[i])

        ls[i] =  stabilf(Sprop[i])
        gs[i] = gsbtf(s[i])*stabideri_both(Dprop[i],Sfprop[i],Cfprop[i])

        gls[i] = gstabilf(Sprop[i])
        glc[i] = gscalelf(Cprop[i])
 
    end

    if both
        logp = -0.5/noisesigma^2*dotitself(res) + sum(ld) + sum(ls) + sum(lc)
    end

    mul!(Gx,M',gld,1,1)
    mul!(Sx,S',gls)
    storeadd!(Sx,Sx,gs)
    mul!(Cx,C',glc)
    storeadd!(Cx,Cx,gc)

    return logp,G

end

function  logpdiffplain(funks,U,arguments,cache)
    Fprop = cache.Fprop
    Dprop = cache.Dprop
    res = cache.residual
    ld = cache.ld
    Nd = length(Dprop)
    x = view(U,1:Nd)
    @unpack plainlf,gplainlf  =  funks

    M = arguments.M;  
    noisesigma = arguments.noisesigma
    y  = arguments.y; F = arguments.F
      
    mul!(Fprop,F,x)
    mul!(Dprop,M,x)
    storesubst!(res,Fprop,y)
 
    #=Threads.@threads=# for i = 1:Nd
        ld[i] =  plainlf(Dprop[i]) 
    end

    logp =  -0.5/noisesigma^2*dotitself(res)
    logp = logp + sum(ld);

    return logp

end

function  logpdiffgradiplain(funks,U,arguments,cache;both=false)
    Fprop = cache.Fprop
    Dprop = cache.Dprop
    res = cache.residual
    @unpack gld, glc, ld, lc, gc = cache
    Nd = length(Dprop)
    G = cache.gradiprop
    Gx = view(G,1:Nd)
    x = view(U,1:Nd)
    logp = 0.0
    @unpack  plainlf,gplainlf  = funks

    M = arguments.M;   
    noisesigma = arguments.noisesigma
    y  = arguments.y; F = arguments.F
       
    mul!(Fprop,F,x)
    mul!(Dprop,M,x)
    storesubst!(res,Fprop,y)   

    mul!(Gx,F',-((res)/noisesigma.^2))
   
    #=Threads.@threads=# for i = 1:Nd
        gld[i] =  gplainlf(Dprop[i])
    end

    if both
        for i = 1:Nd
            ld[i] =  plainlf(Dprop[i])   
        end
        logp = -0.5/noisesigma^2*dotitself(res) + sum(ld) 
    end

    mul!(Gx,M',gld,1,1)

    return logp,G

end

function expsquared(xi,l,s)
    N = length(xi)
    C = zeros(N,N)
    for i = 1:N
        for j = 1:i
            C[i,j] = s^2*exp(-(xi[i]-xi[j])^2/l^2)
            C[j,i] = C[i,j]
        end
    end
    return C + I(N)*1e-9
end

include("1dapprox.jl")

function interp_help1d()
    if !@isdefined ip1d
        
        return polator1d(transition=true)
  
    else
        return ip1d
    end

end

const ip1d = interp_help1d()
#ip1d = polator1d(transition=true)

# using FiniteDifferences
# function testaa()
#     aa = [0.51,1.0,1.5,1.994]
#     rr = [0, -0.001, 0.08, 0.9, 14,-14.9999999,15.0,15.0000001, 16, 29.9999999, 30.0, 30.0000001, 31,101]
#     #rr = [0.08,29.6,30]
#     #rr = [-37.0,40,0,-0.08,29.6]
#     #rr = [-37.0,40,0]

#     for a  in aa
#         for r in rr
           
#             for s in [0.8,1.0]# [0.001, 0.01, 0.1, 1, 10]
#                 # f(x) = logdensityuni(a,abs(x),ip1d;gamma0=s)
#                 # g(x) = logdensityunideri(a,x,ip1d;gamma0=s)
#                 # ad = g(r)
#                 # nd = central_fdm(3,1;max_range=1e-5)(f,r)

#                 # f(x) = logdensityuni(x,r,ip1d;gamma0=s)
#                 # g(x) = logdensityunideristabi(x,r,ip1d;gamma0=s)
#                 # ad = g(a)
#                 # nd = central_fdm(3,1;max_range=1e-5)(f,a)

#                 # f(x) = logdensityuni(a,r,ip1d;gamma0=x)
#                 # g(x) = logdensityunideriscale(a,r,ip1d;gamma0=x)
#                 # ad = g(s)
#                 # nd = central_fdm(3,1;max_range=1e-5)(f,s)
                
#                 @show a,r,s, nd-ad
#             end
#         end
#     end

# end

# testaa()


#stop("Stop")


linspace(x,y,z) = range(x,stop=y,length=z)

Random.seed!(4)

N = 120
Nbig = 500
xi = Vector(range(-0.0,stop=1.0,length=N));  # dx = x[2]-x[1];
measxi = xi[1:2:end] ;
Nmeas = length(measxi)
xibig = Vector(range(-0.0,stop=1.0,length=Nbig));

triangle(x) = 1.0*(x+1)*(x<=0)*(x>=-1) + 1.0*(-x+1)*(x>0)*(x<=1);
heavi(x) = 1.0*(x>=0);
ff(x) = 0+ 0*exp(-60*abs(x-0.2)) - 0*exp(-180*abs(x-0.8)) + 1*heavi(10*(x-0.7))*heavi(-x+0.9) + 0*heavi(x-0.3)*heavi(-x+0.6);
sigma = 0.02;  # Standard deviation of the measurement noise.
c = 50;
kernel(x,y) =   c/2*exp(-c*abs((x-y)));
#ffc(x) = quadgk(y ->  ff(y)*kernel(x,y), 0, 1, rtol=1e-7)[1]

gixbig = findall((xibig .>= 0.4) .&& (xibig .<= 0.6))
Cbig = expsquared(xibig[gixbig],0.04,0.1) 
gtp = cholesky(Cbig).L*randn(length(gixbig)) .+ 0.35

gixbig2 = findall((xibig .>= 0.1) .&& (xibig .<= 0.3))
Cbig2 = expsquared(xibig[gixbig2],0.1,0.2) 
gtp2 = cholesky(Cbig2).L*randn(length(gixbig2)) .- 0.2



Fbig =  measurementmatrix(xibig,measxi,kernel); # Theory matrix.
F =  measurementmatrix(xi,measxi,kernel);
gtbig = ff.(xibig);
gtbig[gixbig] = gtp
gtbig[gixbig2] = gtp2

gt = LinearInterpolation((xibig,), gtbig)(xi)

# ffc.(measxi); 
gtc = Fbig*gtbig
# meas = gtc .+ sigma.*randn(size(measxi)); # Simulate the measurements.

x0 = randn(N)
x1 = abs.(randn(N))

D = difference1(xi);


# qq = D\rand(AlphaStable(α=1.0,scale=100.0),N)
# sr =  sctf.(qq)
# w = zeros(N)
# for i = 1:N
#     w[i] = rand(AlphaStable(α=1.9,scale=sr[i]))
# end

# qq = D\rand(AlphaStable(α=1.0,scale=1.0),N)
# sr =  sbtf.(qq)
# w = zeros(N)
# for i = 1:N
#     w[i] = rand(AlphaStable(α=sr[i],scale=0.1))
# end

# qs = D\rand(AlphaStable(α=stabifield_a,scale=stabifield_gamma),N)
# qc = D\rand(AlphaStable(α=scalefield_a,scale=scalefield_gamma),N)
# s =  sbtf.(qs)
# c =  sctf.(qc)
# w = zeros(N)
# for i = 1:N
#     w[i] = rand(AlphaStable(α=s[i],scale=c[i]))
# end
#Xc = [(D\w);qs;qc]

#meas = F*(D\w) + randn(N)*sigma
meas = gtc + randn(Nmeas)*sigma

tall0 = "deconvot/muut.mat"
matwrite(tall0,Dict("y"=>meas,"gtbig"=>gtbig,"gtc"=>gtc))

arg = (F=F,M=D,y=meas,noisesigma=sigma,S=copy(D),C=copy(D))

# Both
cacB = (Dprop=D*x0,Fprop=F*x0,ld=similar(D*x0), lc =zeros(N), ls=similar(D*x0),gld=similar(D*x0), gs = zeros(N), gc = zeros(N), glc=similar(D*x0), gls = zeros(N), Sprop =zeros(N),  Sfprop =zeros(N),   Cfprop =zeros(N),  Cprop =zeros(N), gradiprop = zeros(3*N), residual=similar(meas))
cacB2 = deepcopy(cacB)
# Either
cacE = (Dprop=D*x0,Fprop=F*x0,ld=similar(D*x0), lc =zeros(N), ls=similar(D*x0),gld=similar(D*x0), gs = zeros(N), gc = zeros(N), glc=similar(D*x0), gls = zeros(N), Sprop =zeros(N),  Sfprop =zeros(N),   Cfprop =zeros(N),  Cprop =zeros(N), gradiprop = zeros(2*N), residual=similar(meas))
cacE2 = deepcopy(cacE)
# Neither
cacN = (Dprop=D*x0,Fprop=F*x0,ld=similar(D*x0), lc =zeros(N), ls=similar(D*x0),gld=similar(D*x0), gs = zeros(N), gc = zeros(N), glc=similar(D*x0), gls = zeros(N), Sprop =zeros(N),  Sfprop =zeros(N),   Cfprop =zeros(N),  Cprop =zeros(N), gradiprop = zeros(N), residual=similar(meas))
cacN2 = deepcopy(cacN)

arg2 = deepcopy(arg); 

#stabifield_a = 1.9
#scalefield_gamma = 0.05

# scalefield_a = 1.99
# scalefield_gamma = 2.0

#condiscale_a = 1.99
#condistabi_gamma = 0.05

#scaplain = 0.02
#staplain = 1.5

#stop()

 for scale ∈  [0.001,  0.01, 0.05, 0.1] #[0.001, 0.01, 0.05, 0.1, 1.0, 5]
     for stabi ∈ [0.51, 0.8, 1.4, 1.7 , 1.9]# [0.55, 0.75, 1.0, 1.2, 1.4, 1.6, 1.8, 1.9, 1.99, 1.999]

        global dataiter = Dict("resi"=>Vector{Float64}(),"logd"=>Vector{Float64}(),"gnorm"=>Vector{Float64}())
        afun(iter) = cb(iter,gt,dataiter)

        stabifield_gamma =  -2.0#scale#3.9818909324060696# scale
        stabifield_a  =  -2.0#stabi#1.136161964126856# stabi
        condistabi_gamma = -2.0#0.01# 0.0506916875512813# condi

        scalefield_gamma =   -2.0#scale#0.3176998433614602#scale
        scalefield_a  =   -2.0#stabi#0.55 #stabi
        condiscale_a =  -2.0#1.9#1.99#condi

        scaplain = -2.0#scale #0.05418189465346412# scale
        staplain =  -2.0#stabi#1.1144493776053677#stabi

        stabifield_gamma = scale#stabis_scale
        stabifield_a  = 1.9#stabis_stabi
        scalefield_gamma =  0.05#scales_scale
        scalefield_a  =  stabi# scales_stabi

        plainlf(x) = logdensityuni(staplain,abs(x),ip1d;gamma0=scaplain)
        gplainlf(x) = logdensityunideri(staplain,x,ip1d;gamma0=scaplain)
        condilf(x,s,c) = logdensityuni(s,abs(x),ip1d;gamma0=c)
        gcondilf(x,s,c) = logdensityunideri(s,x,ip1d;gamma0=c)
        condiscalelf(x,s) = logdensityuni(condiscale_a,abs(x),ip1d;gamma0=s)
        gcondiscalelf(x,s) = logdensityunideri(condiscale_a,x,ip1d;gamma0=s)
        condistabilf(x,s) = logdensityuni(s,abs(x),ip1d;gamma0=condistabi_gamma)
        gcondistabilf(x,s) = logdensityunideri(s,x,ip1d;gamma0=condistabi_gamma)

        stabilf(x) = logdensityuni(stabifield_a,abs(x),ip1d;gamma0=stabifield_gamma)
        gstabilf(x) = logdensityunideri(stabifield_a,x,ip1d;gamma0=stabifield_gamma)
        scalelf(x) = logdensityuni(scalefield_a,abs(x),ip1d;gamma0=scalefield_gamma)
        gscalelf(x) = logdensityunideri(scalefield_a,x,ip1d;gamma0=scalefield_gamma)


        scalederi(x,s) = logdensityunideriscale(condiscale_a,x,ip1d;gamma0=s)
        stabideri(x,s) = logdensityunideristabi(s,x,ip1d;gamma0=condistabi_gamma)

        scalederi_both(x,s,c) = logdensityunideriscale(s,x,ip1d;gamma0=c)
        stabideri_both(x,s,c) = logdensityunideristabi(s,x,ip1d;gamma0=c)

        # sbtf(x) = 0.51 + 1.48*sigmoid(x) 
        # gsbtf(x) = 1.48*dsigmoid(x)

        # sctf(x) = 0.001 + 0.5*sigmoid(x)
        # gsctf(x) = 0.5*dsigmoid(x) 
        
        sbtf(x) = 0.51 + 1.39*sigmoid(x) 
        gsbtf(x) = 1.39*dsigmoid(x)

        sctf(x) = 0.001 + 0.05*sigmoid(x)
        gsctf(x) = 0.05*dsigmoid(x) 


        funks = (plainlf=plainlf,gplainlf=gplainlf,scalederi_both=scalederi_both,stabideri_both=stabideri_both,gcondilf=gcondilf,condilf = condilf,condiscalelf=condiscalelf,gcondiscalelf=gcondiscalelf, scalelf=scalelf,gscalelf=gscalelf,stabilf=stabilf,gstabilf=gstabilf,scalederi=scalederi,sctf=sctf,gsctf=gsctf,sbtf=sbtf,gsbtf=gsbtf,condistabilf=condistabilf,gcondistabilf=gcondistabilf,stabideri=stabideri)


        # error()

        # l3 = logpdiff(ulf,slf,tf,Xc,arg,cac)
        # f(x) = logpdiffboth(funks,x,arg,cac)

        # d2 = logpdiffgradiboth(funks, Xc,arg2,cac2)[2]
        # d1 = gradi(f,Xc;eps=1e-9)


        # targetdiff(w) = -logpdiffstabi(funks,w,arg,cacE)
        # targetdiffgrad(w) = -logpdiffgradistabi(funks, w,arg2,cacE2)[2]
        # global res = optimize(targetdiff, targetdiffgrad, [zeros(N);0*ones(N)],LBFGS(), Optim.Options(allow_f_increases=true,show_trace=false,store_trace=false,extended_trace=true,iterations=1500,callback=afun); inplace = false)
        # global RR = res.minimizer

        # tall = "deconvot/"*"stabi-"*string(stabifield_a)*"-"*string(stabifield_gamma)*"-"*string(condistabi_gamma)*"-"*string(N)*".mat"
        # matwrite(tall,Dict("MAP"=>RR,"logd"=>dataiter["logd"],"resi"=>dataiter["resi"],"gnorm"=>dataiter["gnorm"]))

        # targetdiff(w) = -logpdiffscale(funks,w,arg,cacE)
        # targetdiffgrad(w) = -logpdiffgradiscale(funks, w,arg2,cacE2)[2]
        # global res = optimize(targetdiff, targetdiffgrad, [zeros(N);0*ones(N)],LBFGS(), Optim.Options(allow_f_increases=true,show_trace=false,store_trace=false,extended_trace=true,iterations=1500,callback=afun); inplace = false)
        # global RR = res.minimizer

        # tall = "deconvot/"*"scale-"*string(scalefield_a)*"-"*string(scalefield_gamma)*"-"*string(condiscale_a)*"-"*string(N)*".mat"
        # matwrite(tall,Dict("MAP"=>RR,"logd"=>dataiter["logd"],"resi"=>dataiter["resi"],"gnorm"=>dataiter["gnorm"]))

        targetdiff(w) = -logpdiffboth(funks,w,arg,cacB)
        targetdiffgrad(w) = -logpdiffgradiboth(funks, w,arg2,cacB2)[2]
        global res = optimize(targetdiff, targetdiffgrad, [zeros(N);0*ones(N);0*ones(N)],LBFGS(), Optim.Options(allow_f_increases=true,show_trace=false,store_trace=false,extended_trace=true,iterations=2500,callback=afun); inplace = false)
        global RR = res.minimizer

        tall = "deconvot/"*"both-"*string(scalefield_a)*"-"*string(scalefield_gamma)*"-"*string(stabifield_a)*"-"*string(stabifield_gamma)*"-"*string(N)*".mat"
        matwrite(tall,Dict("MAP"=>RR,"logd"=>dataiter["logd"],"resi"=>dataiter["resi"],"gnorm"=>dataiter["gnorm"]))


        # targetdiff(w) = -logpdiffplain(funks,w,arg,cacN)
        # targetdiffgrad(w) = -logpdiffgradiplain(funks, w,arg2,cacN2)[2]
        # global res = optimize(targetdiff, targetdiffgrad, zeros(N),LBFGS(), Optim.Options(allow_f_increases=true,show_trace=false,store_trace=false,extended_trace=true,iterations=1500,callback=afun); inplace = false)
        # global RR = res.minimizer

        # tall = "deconvot/"*"plain-"*string(staplain)*"-"*string(scaplain)*"-"*string(N)*".mat"
        # matwrite(tall,Dict("MAP"=>RR,"logd"=>dataiter["logd"],"resi"=>dataiter["resi"],"gnorm"=>dataiter["gnorm"]))

    end
end


