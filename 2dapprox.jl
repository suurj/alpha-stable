using SparseArrays
using SpecialFunctions
using MAT
using LinearAlgebra
using Interpolations

cd(@__DIR__)

include("boundaryspline.jl")

stop(text="Stop.") = throw(StopException(text))

struct StopException{T}
    S::T
end

function Base.showerror(io::IO, ex::StopException, bt; backtrace=true)
    Base.with_output_color(get(io, :color, false) ? :green : :nothing, io) do io
        showerror(io, ex.S)
    end
end


function hanta(alpha,r0;gamma0=1.0)
    # r = r0/gamma0
    # l0 = log(r/(2))*(-alpha-1) +2*loggamma((alpha+2)/2.0)+ log(abs(sin(alpha*pi/2.0)))
    # l0g = -(alpha+1)/r 
    # li = 0.0
    # lig = 0.0
    # for n = 2:4
    #     lcur = log(r/(2))*(-n*alpha-1) +2*loggamma((n*alpha+2.0)/2.0)+  -loggamma(n+1) #+ log(abs((sin(n*alpha*pi/2.0))))
    #     termcur = sin(n*alpha*pi/2.0)*(-1)^(n+1)
    #     stermcur = sign(termcur)
    #     lcur = lcur + log(stermcur*termcur)
    #     q = exp(lcur-l0)*stermcur
    #     li = li + q
    #     lig = lig + q*((1/r)*(-n*alpha-1) -l0g)
    #     #println(lcur)
    # end
    # #println(li)
    

    r = r0/gamma0
    l0 = log(r/(2))*(-alpha-1) +2*loggamma((alpha+2)/2.0)+ log(abs(sin(alpha*pi/2.0)))
    l0g = -(alpha+1)/r 
    li = 0.0
    lig = 0.0
    for n = 2:4
        lcur = log(r/(2))*(-n*alpha-1) +2*loggamma((n*alpha+2.0)/2.0)+  -loggamma(n+1) #+ log(abs((sin(n*alpha*pi/2.0))))
        termcur = sin(n*alpha*pi/2.0)*(-1)^(n+1)
        stermcur = sign(termcur)
        lcur = lcur + log(stermcur*termcur)
        q = exp(lcur-l0)*stermcur
        li = li + q
        lig = lig + q*((1/r)*(-n*alpha-1) -l0g)
    end
    return l0 + log1p(li)- log(r ) - 1.8378770664093453 - 1.1447298858494002 - 2*log(gamma0), l0g + lig/(1+li)- 1/r 

end



function logorigo(alpha,r0,gamma0)

    if ( r0 == 0.0)
        return loggamma(2/alpha) - log(alpha*2*pi) - 2*log(gamma0),0.0
    end
    
    r0a = abs(r0)
    r = r0a/gamma0
    logr = log(r)
    log2 = log(2)

    #l = gamma(2/alpha)*(r/(2))

    l0 = loggamma(2/alpha)+ logr - log2
    li = 0.0

    l0g = 1/r0a
    lg = 0.0

    for k = 1:4
        #l = l + (-1)^k*gamma((2*k+2)/alpha)/(factorial(k)*gamma((2*k+2)/2))*(r/(2))^(2*k+2-1)

        lcur =  loggamma((2*k+2)/alpha)- logfactorial(k) - loggamma((2*k+2)/2) + (2*k+2-1)*(logr - log2)
        stermcur = sign((-1)^k)
        q = exp(lcur-l0)*stermcur
        li = li + q
        lg = lg + q*((2*k+2-1)*l0g- l0g)

    end

    #l = l*2/(alpha*gamma0)
    #l = l/(2*pi)*r0^(-1)

    l0 = l0 + log1p(li) - log(pi) - log(alpha*gamma0) - log(r0a)

    lg = (lg/(li+1))*sign(r0)

    return l0,lg# log(l)

end


function uusisiirtyma(interp,v,r,bou1,bou2)
    ur = range(bou1,bou2,10)
    Na = length(v)
    Nr = length(ur)
    S = zeros(Na,Nr)

    a(par) = par.k1*(par.x2-par.x1) - (par.y2-par.y1)
    b(par) = -par.k2*(par.x2-par.x1) + (par.y2-par.y1)
    t(par,x) = (x-par.x1)/(par.x2-par.x1)
    f(par,x) = (1-t(par,x))*par.y1+ t(par,x)*par.y2 + t(par,x)*(1-t(par,x))*( (1-t(par,x))*a(par) +t(par,x)*b(par)  )
    # Molemmat ovat Hermiten interpolointia kahden pisteen välillä annetuilla päätepisteiden derivaatoilla.
    function uusi(x,x1,x2,y1,y2,d1,d2)
        h = x2-x1
        s = x -x1
        return (3*h*s^2-2*s^3)/(h^3)*y2 +  (h^3-3*h*s^2+2*s^3)/(h^3)*y1 + s^2*(s-h)/(h^2)*d2 + s*(s-h)^2/(h^2)*d1
    end

    x1 = ur[1]
    x2 = ur[end]

    for j = 1:Na
        y1 = interp(v[j],x1)
        y2,k2 = hanta(v[j],x2)
        #y2,k2 = hanta(v[ir],r[end])
        k1 = Interpolations.gradient(interp,v[j],x1)[2]
        for i = 1:Nr    
           # S[i,j] = log.(interp(v[j],ur[i]))  
            S[j,i] = uusi(ur[i],x1,x2,y1,y2,k1,k2)        
        end
    end

    leftdr = zeros(Na)
    rightdr = zeros(Na)
    topda = zeros(Nr)
    bottomda = zeros(Nr)


    for i = 1:Na
        leftdr[i] = Interpolations.gradient(interp,v[i],ur[1])[2]*Float64(ur.step)
        #rightdr[i] = Interpolations.gradient(interp,v[i],ur[end])[2]
        h =  hanta(v[i],ur[end])
        rightdr[i] = h[2]*Float64(ur.step)
    end
   
    for i = 1:Nr
        topda[i] = Interpolations.gradient(interp,v[1],ur[i])[1]*Float64(v.step)
        bottomda[i] = Interpolations.gradient(interp,v[end],ur[i])[1]*Float64(v.step)
    end

    interu = interpolate(S,BSpline(Cubic(Interpolations.Natural(OnGrid()))))
    interu = Interpolations.scale(interu,v,ur)

    _,_,coef = coefsbderi2(S,leftdr,bottomda,rightdr,topda)

    interu.itp.coefs.parent .= reshape(coef,size(interu.itp.coefs.parent ))

    return interu

end

function splinesiirtyma(interp,interplow,interphigh,vlow,vhigh,bou0,bou00)

    url  = range(bou00,bou0,trunc(Int64,bou0/0.01)+1)

    Nrl = length(url)
    uvl = copy(vlow)
    Nal = length(uvl)
    Sl = zeros(Nal,Nrl)

    leftdrl = zeros(Nal)
    rightdrl = zeros(Nal)
    topdal = zeros(Nrl)
    bottomdal = zeros(Nrl)


    urh  = range(bou00,bou0,trunc(Int64,bou0/0.01)+1)


    Nrh = length(urh)
    uvh = copy(vhigh)
    Nah = length(uvh)
    Sh = zeros(Nah,Nrh)

    leftdrh = zeros(Nah)
    rightdrh = zeros(Nah)
    topdah = zeros(Nrh)
    bottomdah = zeros(Nrh)

    for j = 1:Nal
        for i = 1:Nrl  
           Sl[j,i] = interp(uvl[j],url[i])  
        end
    end
    for j = 1:Nal
        Sl[j,end] = interplow(uvl[j],url[end])  
    end
    

    for i = 1:Nal
        #leftdrl[i] = Interpolations.gradient(interp,uv[i],ur[1])[2]*Float64(ur.step)
        rightdrl[i] = Interpolations.gradient(interplow,uvl[i],url[end])[2]*Float64(url.step)

    end
   
    for i = 1:Nrl
        topdal[i] = Interpolations.gradient(interplow,uvl[1],url[i])[1]*Float64(uvl.step)
        bottomdal[i] = Interpolations.gradient(interphigh,uvl[end],url[i])[1]*Float64(uvl.step)
    end


    for j = 1:Nah
        for i = 1:Nrh    
           Sh[j,i] = interp(uvh[j],urh[i])       
        end
    end
    for j = 1:Nah
        Sh[j,end] = interphigh(uvh[j],urh[end])  
    end

    for i = 1:Nah
        #leftdrh[i] = Interpolations.gradient(interp,uv[i],ur[1])[2]*Float64(ur.step)
        rightdrh[i] = Interpolations.gradient(interphigh,uvh[i],urh[end])[2]*Float64(url.step)

    end
   
    for i = 1:Nrh
        topdah[i] = Interpolations.gradient(interphigh,uvh[1],urh[i])[1]*Float64(uvh.step)
        bottomdah[i] = Interpolations.gradient(interphigh,uvh[end],urh[i])[1]*Float64(uvh.step)
    end

    #### Boundary near zero for series approximation

    # for j = 1:Nal
    #     leftdrl[j] = logorigo(uvl[j], url[1],1.0)[2]*Float64(url.step) #Interpolations.gradient(interp,uv[i],ur[1])[2]*Float64(ur.step)
    #     Sl[j,1] = logorigo(uvl[j], url[1],1.0)[1] # interplow(uvl[j],url[1])  
    # end

    # for j = 1:Nah
    #     leftdrh[j] = logorigo(uvh[j], urh[1],1.0)[2]*Float64(urh.step)  #Interpolations.gradient(interp,uv[i],ur[1])[2]*Float64(ur.step)
    #     Sh[j,1] = logorigo(uvh[j], urh[1],1.0)[1] #interphigh(uvh[j],urh[1])  
    # end

    ####

    interul = interpolate(Sl,BSpline(Cubic(Interpolations.Natural(OnGrid()))))
    interul = Interpolations.scale(interul,uvl,url)

    _,_,coefl = coefsbderi2(Sl,leftdrl,bottomdal,rightdrl,topdal)

    interul.itp.coefs.parent .= reshape(coefl,size(interul.itp.coefs.parent ))


    interuh = interpolate(Sh,BSpline(Cubic(Interpolations.Natural(OnGrid()))))
    interuh = Interpolations.scale(interuh,uvh,urh)

    _,_,coefh = coefsbderi2(Sh,leftdrh,bottomdah,rightdrh,topdah)

    interuh.itp.coefs.parent .= reshape(coefh,size(interuh.itp.coefs.parent ))

    return interul,interuh


end




function deriv(f,x0::T;eps=1e-9) where T
    eps = convert(T,eps)
    d = (f(x0+eps)-f(x0-eps))/(2*eps)
    return d
end

function logdensityderi(alpha,r0,ip;gamma0=1.0)
    interp = ip.interp; interp2 = ip.interp2; interp3 = ip.interp3;  interp4 = ip.interp4; interp5 = ip.interp5; interp6 = ip.interp6
    r = abs(r0/gamma0)

    bou00 = ip.bou00
    bou0 = ip.bou0
    bou1 = ip.bou1
    bou2 = ip.bou2

    #bous1 = ip.bous1
    #bous2 = ip.bous2

    

    if alpha == 2.0
        return (-r/2)/gamma0
    elseif (bou1>=r>=bou0) && (alpha <= 1.995)
        return (Interpolations.gradient( interp,alpha,r))[2]/gamma0*sign(r0)
    # elseif (bous1>=r>=bou0) && (alpha > 1.995)
    #     return (Interpolations.gradient( interp2,alpha,r))[2]/gamma0*sign(r0)
    elseif (bou00<=r<bou0) && (alpha <= 1.995)
        return (Interpolations.gradient( interp3,alpha,r))[2]/gamma0*sign(r0)
    #elseif (bou00<=r<bou0) #&& (alpha > 1.995)
    #    return (Interpolations.gradient( interp6,alpha,r))[2]/gamma0*sign(r0)
    # elseif (0<=r<bou00) 
    #     if ( r0 == 0.0)
    #         return 0.0
    #     end
        
    #     r0a = abs(r0)
    #     logr = log(r)
    #     log2 = log(2)

    #     l0 = loggamma(2/alpha)+ logr - log2
    #     li = 0.0
    
    #     l0g = 1/r0a
    #     lg = 0.0
    
    #     for k = 1:4
   
    #         lcur =  loggamma((2*k+2)/alpha)- logfactorial(k) - loggamma((2*k+2)/2) + (2*k+2-1)*(logr - log2)
    #         stermcur = sign((-1)^k)
    #         q = exp(lcur-l0)*stermcur
    #         li = li + q
    #         lg = lg + q*((2*k+2-1)*l0g- l0g)
    
    #     end
     
    #     lg = (lg/(li+1))*sign(r0)
    
    #     return lg

    elseif (bou2>=r>=bou1) && (alpha <= 1.995)
        return (Interpolations.gradient( interp4,alpha,r))[2]/gamma0*sign(r0)
    #elseif (bous2>=r>=bous1) && (alpha > 1.995)
    #    return (Interpolations.gradient( interp5,alpha,r))[2]/gamma0*sign(r0)
    elseif (r>bou2)  && (alpha <= 1.995)
        # l0 = log(r/(2))*(-alpha-1) +2*loggamma((alpha+2)/2.0)+ log(abs(sin(alpha*pi/2.0)))
        # l0g = -(alpha+1)/r 
        # li = 0.0
        # lig = 0.0
        # for n = 2:4
        #     lcur = log(r/(2))*(-n*alpha-1) +2*loggamma((n*alpha+2.0)/2.0)+  + log(abs((sin(n*alpha*pi/2.0))))-logfactorial(n)
        #     q = exp(lcur-l0)*sign(sin(n*alpha*pi/2.0)*(-1)^(n+1))
        #     li = li + q
        #     lig = lig + q*((1/r)*(-n*alpha-1) -l0g)
        # end
        # return (l0g + lig/(1+li)- 1/r)gamma0

        r = abs(r0/gamma0)
        l0 = log(r/(2))*(-alpha-1) +2*loggamma((alpha+2)/2.0)+ log(sin(alpha*pi/2.0))
        l0g = -(alpha+1)/r 
        li = 0.0
        lig = 0.0
        for n = 2:4
            lcur = log(r/(2))*(-n*alpha-1) +2*loggamma((n*alpha+2.0)/2.0)+  -loggamma(n+1) #+ log(abs((sin(n*alpha*pi/2.0))))
            termcur = sin(n*alpha*pi/2.0)*(-1)^(n+1)
            stermcur = sign(termcur)
            lcur = lcur + log(stermcur*termcur)
            q = exp(lcur-l0)*stermcur
            li = li + q
            lig = lig + q*((1/r)*(-n*alpha-1) -l0g)
        end
        return (l0g + lig/(1+li)- 1/r )*sign(r0)/gamma0

    else 
        error("Alpha not in allowed range")
    end
end

function logdensity(alpha,r0,ip;gamma0=1.0)
    interp = ip.interp; interp2 = ip.interp2; interp3 = ip.interp3;  interp4 = ip.interp4; interp5 = ip.interp5; interp6 = ip.interp6
    r = abs(r0)/gamma0

    # if isinf(r)
    #     return -Inf
    # end

    bou00 = ip.bou00
    bou0 = ip.bou0
    bou1 = ip.bou1
    bou2 = ip.bou2

    #bous1 = ip.bous1
    #bous2 = ip.bous2


    if alpha == 2.0
        return log(1/(2*pi)) +log(1/(2*1^2))  + (-r^2/(4)) - 2*log(gamma0)
    elseif (bou1>=r>=bou0)  && (alpha <= 1.995)
        return (interp(alpha,r)) - 2*log(gamma0)  
    #elseif (bous1>=r>=bou0)  && (alpha >= 1.995)
    #    return (interp2(alpha,r)) - 2*log(gamma0)   
    elseif (bou00<=r<bou0)   && (alpha <= 1.995)
        return (interp3(alpha,r)) - 2*log(gamma0)
    #elseif (bou00<=r<bou0)   #&& (alpha > 1.995)
    #    return (interp6(alpha,r)) - 2*log(gamma0)
    # elseif (0<=r<bou00)
    #     if ( r0 == 0.0)
    #         return loggamma(2/alpha) - log(alpha*2*pi) - 2*log(gamma0)
    #     end
        
    #     r0a = abs(r0)
    #     logr = log(r)
    #     log2 = log(2)
    
    #     l0 = loggamma(2/alpha)+ logr - log2
    #     li = 0.0
     
    #     for k = 1:4
    
    #         lcur =  loggamma((2*k+2)/alpha)- logfactorial(k) - loggamma((2*k+2)/2) + (2*k+2-1)*(logr - log2)
    #         stermcur = sign((-1)^k)
    #         q = exp(lcur-l0)*stermcur
    #         li = li + q
    
    #     end
    
    #     l0 = l0 + log1p(li) - log(pi) - log(alpha*gamma0) - log(r0a)
    
    #     return l0

    elseif (bou2>=r>=bou1)  && (alpha <= 1.995)
        return (interp4(alpha,r)) - 2*log(gamma0) 
    #elseif (bous2>=r>=bous1)  && (alpha > 1.995)
    #    return (interp5(alpha,r)) - 2*log(gamma0)  
    elseif  (r>bou2)  && (alpha <= 1.995)
        # l0 = log(r/(2))*(-alpha-1) +2*loggamma((alpha+2)/2.0)+ log(abs(sin(alpha*pi/2.0)))
        # li = 0.0
        # for n = 2:4
        #     lcur = log(r/(2))*(-n*alpha-1) +2*loggamma((n*alpha+2.0)/2.0)+  + log(abs((sin(n*alpha*pi/2.0))))-logfactorial(n)
        #     li = li + exp(lcur-l0)*sign(sin(n*alpha*pi/2.0)*(-1)^(n+1))
        #     #println(lcur)
        # end
        # #println(li)
        # return l0 + log1p(li)- log(r ) - 1.8378770664093453 - 1.1447298858494002 - 2*log(gamma0)  #    log(2*pi) - log(pi)

        l0 = log(r/2.0)*(-alpha-1) +2*loggamma((alpha+2)/2.0)+ log(sin(alpha*pi/2.0))
        #l0g = -(alpha+1)/r 
        li = 0.0
        #lig = 0.0
        for n = 2:4
            lcur = log(r/2.0)*(-n*alpha-1) +2*loggamma((n*alpha+2.0)/2.0)+  -loggamma(n+1) #+ log(abs((sin(n*alpha*pi/2.0))))
            termcur = sin(n*alpha*pi/2.0)*(-1)^(n+1)
            stermcur = sign(termcur)
            lcur = lcur + log(stermcur*termcur)
            q = exp(lcur-l0)*stermcur
            li = li + q
            #lig = lig + q*((1/r)*(-n*alpha-1) -l0g)
        end
        ret =  l0 + log1p(li)- log(r ) - 1.8378770664093453 - 1.1447298858494002 - 2*log(gamma0)
        return ret

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



function polator(;transition=false)

    
    bou1 = 29.6
    bou2 = 30.0

    bous1 = 14.6
    bous2 = 15.0

    # q05=matread("hila_05_10095.mat")["M"]
    # q101=matread("hila_101_1999.mat")["M"]
    # q2=matread("hila_19995_2.mat")["M"]
    q = matread("hila_05_20.mat")["M"]
    r = 0.0:0.01:30
    v = 0.5:0.0005:(2.0)
    lq = log.(q)

    q2 = matread("hila_1995.mat")["M"]
    r2 = 0.0:0.01:15
    v2 = range(1.995,stop=2.0,length=3000)
    lq2 = log.(q2)

    # bou0 = 0.08
    # q3 = matread("hila_nolla_0.1.mat")["M"]
    # lq3 = log.(q3)
    # v3 = range(0.5,stop=2.0,length=3000)
    # r3 = range(0,stop=0.1,length=500)

    bou0 = 0.9
    bou00 = 0.00
    q3 = matread("hila_nolla_1.0.mat")["M"]
    lq3 = log.(q3)
    v3 = range(0.5,stop=2.0,length=3000)
    r3 = range(0,stop=1.0,length=500)



    #interp = interpolate((v,r),qtot,Gridded(Linear()))
    #grid = RectangleGrid(r,v)   
    #interp=Spline2D(v, r, qtot; kx=1, ky=1, s=0.0)
    #itp(w)=interpolate(grid,qtot,w)
    #interp = CubicSplineInterpolation((v,r),lq)
    #interp2 = CubicSplineInterpolation((v2,r2),lq2)
    #interp3 = CubicSplineInterpolation((v3,r3),q3)
    interp = interpolate(lq,BSpline(Cubic(Interpolations.Natural(OnGrid())))); interp = Interpolations.scale(interp,v,r)
    interp2 = interpolate(lq2,BSpline(Cubic(Interpolations.Natural(OnGrid())))); interp2 = Interpolations.scale(interp2,v2,r2)
    interp3 = interpolate(lq3,BSpline(Cubic(Interpolations.Flat(OnGrid())))); interp3 = Interpolations.scale(interp3,v3,r3)
    interp4 = deepcopy(interp)
    interp5 = deepcopy(interp2)
    interp6 = deepcopy(interp3)

    # uq=matread("hila1d_tasa_05_20.mat")["M"]
    # uv = 0.5:0.0005:2.0
    # ur = 0.0:0.01:30  
    # ulq = log.(uq)


    # uq2=matread("hila1d_1995.mat")["M"]
    # uv2  = range(1.995,stop=2.0,length=3000)
    # ur2 = 0.0:0.01:30
    # ulq2 = log.(uq2)

    # uq3=matread("hila1d_nolla.mat")["M"]
    # uv3 = range(0.5,stop=2.0,length=3000)
    # ur3 = range(0,stop=0.02,length=500)
    # ulq3 = log.(uq3)

    # uinterp = CubicSplineInterpolation((uv,ur),ulq)
    # uinterp2 = CubicSplineInterpolation((uv2,ur2),ulq2)
    # #uinterp3 = CubicSplineInterpolation((uv3,ur3),ulq3)
    # uinterp3 = interpolate(ulq3,BSpline(Cubic(Interpolations.Flat(OnGrid())))); uinterp3 = scale(uinterp3,uv3,ur3)

    # if transition
    #     # lqn,lqn2 = siirtyma(uinterp,uinterp2,v,v2,r,r2,lq,lq2)
    #     # uinterp = CubicSplineInterpolation((v,r),lqn)
    #     # uinterp2 = CubicSplineInterpolation((v2,r2),lqn2)

    #     lqn,lqn2 = siirtyma(interp,interp2,v,v2,r,r2,lq,lq2)
        interp = CubicSplineInterpolation((v,r),lqn)
    #     interp2 = CubicSplineInterpolation((v2,r2),lqn2)

    # end

    if transition
        interp4 = uusisiirtyma(interp,v,r,bou1,bou2)
        interp5 = uusisiirtyma(interp2,v2,r2,bous1,bous2)
        interp3,interp6 = splinesiirtyma(interp3,interp,interp2,v,v2,bou0,bou00)
    end

    ip = (interp = interp, interp2 = interp2, interp3 = interp3, interp4=interp4, interp5=interp5, interp6 = interp6, bou00 = bou00, bou0=bou0, bou1 = bou1, bou2 = bou2, bous1 = bous1, bous2 = bous2)
    #ip1d = (uinterp = uinterp, uinterp2 = uinterp2, uinterp3 = uinterp3)

    return ip#, ip1d

end


ip = polator(transition=true)

