using Interpolations
using MAT
using SpecialFunctions

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


function hanta1d(alpha,r0,gamma0=1.0)

    logr = log(r0)
    c = -(log(pi) + logr)

    # l0 = loggamma(alpha+1) + log(-cos(pi*(alpha+1)/2)) -(alpha+1)*logr

    # li = 0.0
    # for n = 2:3
    #     lcur = -logfactorial(n) -logr*(alpha*n+1) + log(abs(cos(pi*(alpha*n+1)/2))) +loggamma(alpha*n+1)
    #     li = li + exp(lcur-l0)
    # end

    r = r0

    l0 =  loggamma(alpha+1) + log(sin(pi*(alpha)/2)) -(alpha)*log(r)
    l0g = -(alpha+1)*1/r
    li = 0.0
    lig = 0.0
    for n = 2:4
        lcur = -loggamma(n+1) -logr*(alpha*n)  +loggamma(alpha*n+1)
        termcur = sin(n*pi*(alpha)/2)*(-1)^(n-1)
        stermcur = sign(termcur)
        lcur = lcur + log(stermcur*termcur)
        q = stermcur*exp(lcur-l0)
        li = li + q
        lig = lig + q*(-l0g + -1/r*(alpha*n+1))
    end
    #println(l0)
    return c+ l0 + log1p(li) -log(gamma0) , (l0g + lig/(1+li))*sign(r0)/gamma0

end



function uusisiirtyma1d(interp,v,r,q,bou1,bou2)
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
        y2,k2 = hanta1d(v[j],x2)
        #y2,k2 = hanta1d(v[ir],r[end])
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
        h =  hanta1d(v[i],ur[end])
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

function splinesiirtyma1d(interp,interplow,interphigh,vlow,vhigh,bou0)

    T = 5
    # q = copy(q)
    # q = q[:,1:T:end]

    url  = range(0,bou0,100)

    Nrl = length(url)
    uvl = copy(vlow)
    Nal = length(uvl)
    Sl = zeros(Nal,Nrl)

    leftdrl = zeros(Nal)
    rightdrl = zeros(Nal)
    topdal = zeros(Nrl)
    bottomdal = zeros(Nrl)



    urh = range(0,bou0,100)

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
           #Sl[j,i] = q[j,i] 
           Sl[j,i] = interp(uvl[j],url[i])  
        end
    end
    for j = 1:Nal
        #Sl[j,i] = q[j,i] 
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
        #Sl[j,i] = q[j,i] 
        Sh[j,end] = interphigh(uvh[j],urh[end])  
     end

    for i = 1:Nah
        #leftdr[i] = Interpolations.gradient(interp,uv[i],ur[1])[2]*Float64(ur.step)
        rightdrh[i] = Interpolations.gradient(interphigh,uvh[i],urh[end])[2]*Float64(url.step)

    end
   
    for i = 1:Nrh
        topdah[i] = Interpolations.gradient(interphigh,uvh[1],urh[i])[1]*Float64(uvh.step)
        bottomdah[i] = Interpolations.gradient(interphigh,uvh[end],urh[i])[1]*Float64(uvh.step)
    end

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

function splinesiirtymaloppu1d(interp,interplow,interphigh,vlow,vhigh,bou1, bou2)

    url  = range(bou1,bou2,trunc(Int64,(bou2-bou1)/0.01)+1)

    Nrl = length(url)
    uvl = copy(vlow)
    Nal = length(uvl)
    Sl = zeros(Nal,Nrl)

    leftdrl = zeros(Nal)
    rightdrl = zeros(Nal)
    topdal = zeros(Nrl)
    bottomdal = zeros(Nrl)

    urh = range(bou1,bou2,trunc(Int64,(bou2-bou1)/0.01)+1)

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
           Sl[j,i] = interplow(uvl[j],url[i])  
        end
    end

    for j = 1:Nal
        Sl[j,end] = hanta1d(uvl[j],url[end])[1]
    end
    

    for i = 1:Nal
        leftdrl[i] = Interpolations.gradient(interplow,uvl[i],url[1])[2]*Float64(url.step)
        rightdrl[i] = hanta1d(uvl[i],url[end])[2]*Float64(urh.step)

    end
   
    for i = 1:Nrl
        topdal[i] = Interpolations.gradient(interplow,uvl[1],url[i])[1]*Float64(uvl.step)
        bottomdal[i] = Interpolations.gradient(interphigh,uvl[end],url[i])[1]*Float64(uvl.step)
    end


    for j = 1:Nah
        for i = 1:Nrh    
           Sh[j,i] = interphigh(uvh[j],urh[i])       
        end
    end

    for j = 1:Nah
        Sh[j,end] = hanta1d(uvh[j],urh[end])[1]
    end

    for i = 1:Nah
        leftdrh[i] = Interpolations.gradient(interphigh,uvh[i],urh[1])[2]*Float64(urh.step)
        rightdrh[i] = hanta1d(uvh[i],urh[end])[2]*Float64(urh.step)

    end
   
    for i = 1:Nrh
        topdah[i] = Interpolations.gradient(interphigh,uvh[1],urh[i])[1]*Float64(uvh.step)
        bottomdah[i] = Interpolations.gradient(interphigh,uvh[end],urh[i])[1]*Float64(uvh.step)
    end

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

#qn,qn2 = siirtyma(interp,interp2,v,v2,r,r2,q,q2)
#interp = CubicSplineInterpolation((v,r),qn)
#interp2 = CubicSplineInterpolation((v2,r2),qn2)

# mones = 2900
# osanen = CubicSplineInterpolation(r,q[mones,:])


function deriv(f,x0::T;eps=1e-9) where T
    eps = convert(T,eps)
    d = (f(x0+eps)-f(x0-eps))/(2*eps)
    return d
end


function logdensityunideri(alpha,r0,ip;gamma0=1.0)
    interp = ip.uinterp; interp2 = ip.uinterp2; interp3 = ip.uinterp3;  interp4 = ip.uinterp4; interp5 = ip.uinterp5; interp6 = ip.uinterp6
    r = abs(r0/gamma0)

    bou0 = ip.bou0
    bou1 = ip.bou1
    bou2 = ip.bou2

    if alpha == 2.0
        return  sign(r0)/gamma0*(-r/(2)) 
    elseif (bou1>=r>=bou0)  && (alpha <= 1.995)
        return sign(r0)/gamma0*(Interpolations.gradient(interp,alpha,r))[2]# -log(gamma0)
    #elseif (bou1>=r>=bou0)  && (alpha > 1.995)
    #    return sign(r0)/gamma0*(Interpolations.gradient(interp2,alpha,r))[2]# -log(gamma0)
    elseif (0<=r<bou0) && (alpha <= 1.995)
        return sign(r0)/gamma0*(Interpolations.gradient(interp3,alpha,r))[2]
    #elseif (0<=r<bou0) && (alpha > 1.995)
    #    return sign(r0)/gamma0*(Interpolations.gradient(interp6,alpha,r))[2]
    elseif (bou2>=r>=bou1)  && (alpha <= 1.995)
        return sign(r0)/gamma0*(Interpolations.gradient(interp4,alpha,r))[2]
    #elseif (bou2>=r>=bou1)  && (alpha > 1.995)
    #    return sign(r0)/gamma0*(Interpolations.gradient(interp5,alpha,r))[2]
    elseif (r>bou2) && (alpha <= 1.995)

        logr = log(r)
        #c = -(log(pi) + logr)

        l0 =  loggamma(alpha+1) + log(sin(pi*(alpha)/2)) -(alpha)*log(r)
        l0g = -(alpha+1)*1/r
        li = 0.0
        lig = 0.0
        for n = 2:4
            lcur = -loggamma(n+1) -logr*(alpha*n)  +loggamma(alpha*n+1)
            termcur = sin(n*pi*(alpha)/2)*(-1)^(n-1)
            stermcur = sign(termcur)
            lcur = lcur + log(stermcur*termcur)
            q = stermcur*exp(lcur-l0)
            li = li + q
            lig = lig + q*(-l0g + -1/r*(alpha*n+1))
        end
        #println(l0)
        return  (l0g + lig/(1+li))*sign(r0)/gamma0
    else 
        error("Alpha not in allowed range")
    end
end

#(-1)^n/factorial(n)*(1/r)^(alpha*n+1)*cos(pi*(alpha*n+1)/2)*gamma(alpha*n+1)
function logdensityuni(alpha,r0,ip;gamma0=1.0)
    interp = ip.uinterp; interp2 = ip.uinterp2; interp3 = ip.uinterp3; interp4 = ip.uinterp4; interp5 = ip.uinterp5; interp6 = ip.uinterp6
    r = abs(r0/gamma0)

    bou0 = ip.bou0
    bou1 = ip.bou1
    bou2 = ip.bou2

    if alpha == 2.0
        si = sqrt(2*gamma0^2)
        return -0.5*log((2*pi)) -log(si)  + (-r^2/(2*si^2)) 
    elseif (bou1>=r>=bou0)  && (alpha <= 1.995)
        return (interp(alpha,r)) -log(gamma0)
    #elseif (bou1>=r>=bou0)  && (alpha > 1.995)
    #    return (interp2(alpha,r)) -log(gamma0)
    elseif (0<=r<bou0)  && (alpha <= 1.995)
        return (interp3(alpha,r)) -log(gamma0)
    #elseif (0<=r<bou0)  && (alpha > 1.995)
    #    return (interp6(alpha,r)) -log(gamma0)
    elseif (bou2>=r>=bou1)  && (alpha <= 1.995)
        return (interp4(alpha,r)) -log(gamma0)
    #elseif (bou2>=r>=bou1)  && (alpha > 1.995)
    #    return (interp5(alpha,r)) -log(gamma0)
    elseif (r>bou2) && (alpha <= 1.995)
        # logr = log(r)
        # l0 = loggamma(alpha+1) + log(-cos(pi*(alpha+1)/2)) -(alpha+1)*logr
        # li = 0.0
        # for n = 2:3
        #     lcur = -logfactorial(n) -logr*(alpha*n+1) + log(abs(cos(pi*(alpha*n+1)/2))) +loggamma(alpha*n+1)
        #     li = li + exp(lcur-l0)
        # end
        # #println(li)
        # return l0 + log1p(li) -log(gamma0) - 1.1447298858494002 #log(pi)

        logr = log(r)
        c = -(log(pi) + logr)

        l0 =  loggamma(alpha+1) + log(sin(pi*(alpha)/2)) -(alpha)*log(r)
        li = 0.0
        #lig = 0.0
        for n = 2:4
            lcur = -loggamma(n+1) -logr*(alpha*n)  +loggamma(alpha*n+1)
            termcur = sin(n*pi*(alpha)/2)*(-1)^(n-1)
            stermcur = sign(termcur)
            lcur = lcur + log(stermcur*termcur)
            q = stermcur*exp(lcur-l0)
            li = li + q         
        end
        return c+ l0 + log1p(li) -log(gamma0) 
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


function polator1d(;transition=false)

   
    
    bou1 = 29.6
    bou2 = 30.0

    q=matread("hila1d_tasa_05_20.mat")["M"]
    v = 0.5:0.0005:2.0
    r = 0.0:0.01:30  
    lq = log.(q)

    q2=matread("hila1d_1995.mat")["M"]
    v2  = range(1.995,stop=2.0,length=3000)
    r2 = 0.0:0.01:30
    lq2 = log.(q2)

    # bou0 = 0.08
    # q3=matread("hila1d_nolla_0.1.mat")["M"]
    # v3 = range(0.5,stop=2.0,length=3000)
    # r3 = range(0,stop=0.1,length=500)
    # lq3 = log.(q3)

    bou0 = 0.9
    q3=matread("hila1d_nolla_1.0.mat")["M"]
    v3 = range(0.5,stop=2.0,length=3000)
    r3 = range(0,stop=1.0,length=500)
    lq3 = log.(q3)

    #uinterp = CubicSplineInterpolation((v,r),lq)
    #uinterp2 = CubicSplineInterpolation((v2,r2),lq2)
    #uinterp3 = CubicSplineInterpolation((v3,r3),lq3)
    uinterp = interpolate(lq,BSpline(Cubic(Interpolations.Natural(OnGrid())))); uinterp = Interpolations.scale(uinterp,v,r)
    uinterp2 = interpolate(lq2,BSpline(Cubic(Interpolations.Natural(OnGrid())))); uinterp2 = Interpolations.scale(uinterp2,v2,r2)
    uinterp3 = interpolate(lq3,BSpline(Cubic(Interpolations.Flat(OnGrid())))); uinterp3 = Interpolations.scale(uinterp3,v3,r3)
    uinterp4 = deepcopy(uinterp)
    uinterp5 = deepcopy(uinterp2)
    uinterp6 = deepcopy(uinterp3)

    # if transition
    #     lqn,lqn2 = siirtyma1d(uinterp,uinterp2,v,v2,r,r2,lq,lq2)
    #     uinterp = CubicSplineInterpolation((v,r),lqn)
    #     uinterp2 = CubicSplineInterpolation((v2,r2),lqn2)

    # end

     if transition
        uinterp4 = uusisiirtyma1d(uinterp,v,r,lq,bou1,bou2)
        uinterp5 = uusisiirtyma1d(uinterp2,v2,r2,lq2,bou1,bou2)
        #uinterp4,uinterp5 = splinesiirtymaloppu1d(uinterp,uinterp,uinterp2,v,v2,bou1,bou2)

        uinterp3,uinterp6 = splinesiirtyma1d(uinterp3,uinterp,uinterp2,v,v2,bou0)
    end


    ip = (uinterp = uinterp, uinterp2 = uinterp2, uinterp3 = uinterp3, uinterp4=uinterp4, uinterp5=uinterp5,  uinterp6 = uinterp6, bou0 = bou0, bou1 = bou1, bou2 = bou2)

    return ip

end


ip1d = polator1d(transition=true)
