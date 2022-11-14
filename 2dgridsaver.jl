using Interpolations: length
using SpecialFunctions
using QuadGK
using PyPlot
using Statistics
using AlphaStableDistributions
using Interpolations
using LinearAlgebra
using MAT
using ProgressBars


@fastmath @inline function coeffi(k::Int64,alpha::T,d::Int64) where T
    return (-1.0)^(k+1)*gamma((k*alpha+2.0)/2.0)*gamma((k*alpha+d)/2.0)*sin(k*alpha*pi/2.0)/factorial(k)
end

function tail(alpha::T,gamma0::T,d::Int64,r::T) where T

    return gamma(d/2)/(2*pi^(d/2))*r^(1-d) * 1/(pi*gamma0*gamma(d/2))*((r/(2*gamma0))^(-alpha-1)*coeffi(1,alpha,d) +(r/(2*gamma0))^(-2*alpha-1)*coeffi(2,alpha,d) +(r/(2*gamma0))^(-3*alpha-1)*coeffi(3,alpha,d) +(r/(2*gamma0))^(-4*alpha-1)*coeffi(4,alpha,d)  )
end    


@inline function pdfcauchy2(r)
    return 1/(2*pi*(1+r^2)^(3/2))
end

@inline function pdfgaussian2(r)
    return  1/(2*pi)*r^(-1)*1/(2*1^2)*r*exp(-r^2/(4*1^2))
end

@inline function inte(r,t,alpha)
    return (r*t)*besselj(0,r*t)*exp(-1^alpha*t^alpha)
end

@inline function pdfplain(r,alpha)
    return 1/(2*pi)*r^(-1)*quadgk(t->inte(r,t,alpha),0,Inf,rtol=1e-10)[1]
end

function pdfm(alpha,r)
    if (r>0)
        if (alpha==1.0)
            return pdfcauchy2(r)
        elseif (alpha==2.0)
            return pdfgaussian2(r)
        else (r>0)           
            return pdfplain(r,alpha)
    end
    else 
        return  gamma(2/alpha)/(alpha*2*pi)
    end
end

 10.0.^(range(-3,stop=1.5,length=2000))
# r = 0.0:0.01:15

v = range(0.5,stop=2.0,length=3000)
r = range(0,stop=0.1,length=500)


Na = length(v)
Nr = length(r)

M = zeros(length(v),length(r))
pb = ProgressBar(1:Na)
pbb = ProgressBar(1:Nr)

d = 2
tv = zeros(length(v))

for i in pb
    
    alpha = v[i]         
    gamma0 = 1.0  
   

    d0 = pdfm(alpha,0.0)
    Threads.@threads for j = 1:Nr
        arvo = pdfm(alpha,r[j])
        M[i,j] = arvo
        #tv[i] = tail(alpha,1.0,2,r[j])
    end
end

#matwrite("hila_nolla.mat", Dict("M" => M))
