using Interpolations
using SpecialFunctions
using QuadGK
using PyPlot
using Statistics
using AlphaStableDistributions
using Interpolations
using LinearAlgebra
using MAT
using ProgressBars
using Roots
using HCubature


#  Nolan's method.
@inline function zeta(alpha::T,beta::T) where T
    return -beta*tan(pi*alpha/2)
end

@inline function theta0(alpha::T,beta::T) where T
    return 1/alpha*atan(beta*tan(pi*alpha/2))
end

@inline function V(alpha::T,beta::T,theta::T) where T
    t0 = theta0(alpha,beta)
    return (cos(alpha*t0))^(1/(alpha-1))*(cos(theta)/(sin(alpha*(theta+t0))))^(alpha/(alpha-1))*cos(alpha*t0+theta*(alpha-1))/cos(theta)
end 

@inline function logV(alpha,beta,theta)
    t0 = theta0(alpha,beta)
    return log(cos(alpha*t0))*(1/(alpha-1))+log(cos(theta)/(sin(alpha*(theta+t0))))*(alpha/(alpha-1))+log(cos(alpha*t0+theta*(alpha-1))/cos(theta))
end

@inline function aux(alpha::T,x::T,z::T) where T
    return (((x-z))^(alpha/(alpha-1)))
end

@inline function logaux(alpha,x,z)
    return log(((x-z)))*(alpha/(alpha-1))
end

@inline function g(alpha::T,beta::T,theta::T,x::T) where T
    z = zeta(alpha,beta)
    return (x-z)^(alpha/(alpha-1))*V(alpha,beta,theta)
end



@inline function h(alpha::T,beta::T,theta::T,x::T) where T
    z = zeta(alpha,beta)
    Vv = V(alpha,beta,theta)
    try 
        a = aux(alpha,x,z)
        ret = Vv*a*exp(-a*Vv)
        #ret = real((Complex(x-z))^(alpha/(alpha-1))*Vv*exp(-Complex(x-z)^(alpha/(alpha-1))*Vv))       
        return ret
    catch
        println(theta,",",x,",",z)
        error()
    end
    return NaN
end



@inline function alphadensity(alpha::T,beta::T,xp::T) where T
    z = zeta(alpha,beta)
    x = xp + z
    t0 = theta0(alpha,beta)
    p2 = convert(T,π)/convert(T,2)
    rajoitus(x) = max(min(x,p2),-t0)

    if ( x> z)   

        r(p) = g(alpha,beta,p,xp) - convert(T,1)
        imax = find_zero(r, (-t0,p2), Bisection())  
        #if (imax+0.001)     
        ra = (-t0,imax,p2)
        #@show ra,imax
        Integraali = quadgk(t->h.((alpha),(beta),(t),(x)),ra...,rtol=1e-10,order=20)[1] 
        #@show Integraali

        #Integraali = hquadrature(t->h.(alpha,beta,t,x),-t0,p2,   rtol=1e-10,initdiv=20)[1]
        #Integraali = quadgk(t->h.((alpha),(beta),(t),(x)),-t0,p2,rtol=1e-10,order=20)[1]
        return alpha/(pi*(x-z)*abs(alpha-1))*Integraali

    elseif (x==z)
        return gamma(1+1/alpha)*cos(t0)/(pi*(1+z^2)^(1/(2*alpha)))

    else
        return alphadensity(alpha,-beta,-xp)
    end
end

function tailap(alpha,r)
    val = 0.0
    for n = 1:15
        val = val + real((-1)^n/factorial(n)*(im/r)^(alpha*n+1)*gamma(alpha*n+1))
    end
    return val/pi
end

function symint(alpha,r,t)
    return exp(-t^alpha)*cos(r*t)
end

function pdfsym(alpha,r)
    return 1/pi*quadgk(t->symint(alpha,r,t),0,Inf,rtol=1e-10)[1]
end

pdfcauchy1(r) = 1/(pi*(1+r^2))
pdfgaussian1(r) = 1/2/sqrt(pi)*exp(-0.25*r^2)

function pdf(alpha,r)
    if (alpha==2.0)
        return pdfgaussian1(r)
    elseif (alpha==1.0)
        return pdfcauchy1(r)
    elseif (abs(alpha-1)<0.2)
        return pdfsym(alpha,r)
        #return Float64(alphadensity(BigFloat(alpha),BigFloat(0.0),BigFloat(r)) )
    else
        #return pdfsym(alpha,r) # Hmm?
        return alphadensity(Float64(alpha),Float64(0.0),Float64(r)) 
    end
end


    
 

    v = 0.5:0.0005:2.0
    r = 0.0:0.01:30 
    
    Na = length(v)
    Nr = length(r)

    M = zeros(length(v),length(r))
    pb = ProgressBar(1:Na)
    pbb = ProgressBar(1:Nr)

    for i in pb
        
        alpha = v[i]         
        gamma0 = 1.0
        
      
        Threads.@threads for j = 1:Nr
            arvo = pdf(alpha,r[j])
            if (arvo == 0)
                error(string(alpha)*","*string(r[j]))
            end
            M[i,j] = arvo

        end
    end

    #matwrite("hila1d_tasa_05_20.mat", Dict("M" => M))
