using Random
using QuadGK
using PyPlot
using Statistics
using Interpolations
using LinearAlgebra
using Optim
using SparseArrays
using ProgressBars

cd( @__DIR__)



@inline  function etaisyys(x,y)
    return sqrt(x^2 + y^2)
end


macro delta()
    return :(1e-5)
end



function  logpisodiffgradi_cauchy(x,args,cache;both=true)
    noisesigma = args.noisesigma
    scale = args.scale; y  = args.y; F = args.F
    bscale = args.bscale
    Dx = args.Dx; Dy = args.Dy
    Db = args.Db

    logp = 0.0

    res = cache.residual
    Dxprop = cache.Dxprop
    Dyprop = cache.Dyprop
    Dbprop = cache.Dbprop
    Fxprop = cache.Fxprop
    G = cache.gradiprop

    #Fxprop .= F*x
    mul!(Fxprop,F,x)
    storesubst!(res,Fxprop,y)
    #res .= Fxprop - y  
    #Dxprop .= Dx*x
    mul!(Dxprop,Dx,x)
    #Dyprop .= Dy*x
    mul!(Dyprop,Dy,x)
    #Dbprop .= Db*x
    mul!(Dbprop,Db,x)

    mul!(G,F',-((res)/noisesigma.^2))

    den = (scale^2 .+ Dyprop.^2 + Dxprop.^2)

    if both
        logp = -0.5/noisesigma^2*dotitself(res) -3/2*sum(log.(den)) - sum(log.(bscale^2 .+ Dbprop.^2))
    end

    mul!(G,-3.0*(Dx)',(Dxprop./den),1,1)
    mul!(G,-3.0*(Dy)',(Dyprop./den),1,1)
    mul!(G,-Db',(2.0*Dbprop./(bscale^2 .+ Dbprop.^2)),1,1)
    #println(maximum(G))

    return logp, G 
end

function  logpisodiffgradi_tv(x,args,cache;both=true)
    noisesigma = args.noisesigma
    scale = args.scale; y  = args.y; F = args.F
    bscale = args.bscale
    Dx = args.Dx; Dy = args.Dy
    Db = args.Db

    logp = 0.0

    res = cache.residual
    Dxprop = cache.Dxprop
    Dyprop = cache.Dyprop
    Dbprop = cache.Dbprop
    Fxprop = cache.Fxprop
    G = cache.gradiprop
    gld = cac.gld
    glb = cac.glb

    #Fxprop .= F*x
    mul!(Fxprop,F,x)
    storesubst!(res,Fxprop,y)
    #res .= Fxprop - y  
    #Dxprop .= Dx*x
    mul!(Dxprop,Dx,x)
    #Dyprop .= Dy*x
    mul!(Dyprop,Dy,x)
    #Dbprop .= Db*x
    mul!(Dbprop,Db,x)

    mul!(G,F',-((res)/noisesigma.^2))

    den = sqrt.( @delta() .+ Dxprop.^2 + Dyprop.^2)

    if both
        logp = -0.5/noisesigma^2*dotitself(res) -scale*sum(sqrt.( @delta() .+ Dxprop.^2 + Dyprop.^2))  -bscale*sum(sqrt.(@delta() .+ Dbprop.^2))
    end

    mul!(G,-scale*(Dx)',(Dxprop./den),1,1)
    mul!(G,-scale*(Dy)',(Dyprop./den),1,1)
    mul!(G,-bscale*Db',(Dbprop./(sqrt.(@delta() .+ Dbprop.^2))),1,1)
    #println(maximum(G))

    return logp, G 
end

function  logpisodiff_tv(x,args,cache)
    noisesigma = args.noisesigma
    scale = args.scale; y  = args.y; F = args.F
    bscale = args.bscale
    Dx = args.Dx; Dy = args.Dy
    Db = args.Db
   

    res = cache.residual
    Dxprop = cache.Dxprop
    Dyprop = cache.Dyprop
    Dbprop = cache.Dbprop
    Fxprop = cache.Fxprop

    mul!(Fxprop,F,x)
    #res .= Fxprop - y  
    storesubst!(res,Fxprop,y)
    #Dxprop .= Dx*x
    mul!(Dxprop,Dx,x)
    #Dyprop .= Dy*x
    mul!(Dyprop,Dy,x)
    #Dbprop .= Db*x
    mul!(Dbprop,Db,x)

    return  -0.5/noisesigma^2*dotitself(res) -scale*sum(sqrt.( @delta() .+ Dxprop.^2 + Dyprop.^2))  -bscale*sum(sqrt.(@delta() .+ Dbprop.^2))

end

function  logpisodiff_gauss(x,args,cache)
    noisesigma = args.noisesigma
    scale = args.scale; y  = args.y; F = args.F
    bscale = args.bscale
    Dx = args.Dx; Dy = args.Dy
    Db = args.Db
   

    res = cache.residual
    Dxprop = cache.Dxprop
    Dyprop = cache.Dyprop
    Dbprop = cache.Dbprop
    Fxprop = cache.Fxprop

    mul!(Fxprop,F,x)
    #res .= Fxprop - y  
    storesubst!(res,Fxprop,y)
    #Dxprop .= Dx*x
    mul!(Dxprop,Dx,x)
    #Dyprop .= Dy*x
    mul!(Dyprop,Dy,x)
    #Dbprop .= Db*x
    mul!(Dbprop,Db,x)

    return  -0.5/noisesigma^2*dotitself(res) -scale*dotitself(Dxprop)  -scale*dotitself(Dyprop)  -bscale*dotitself(Dbprop)

end

function  logpisodiffgradi_gauss(x,args,cache;both=true)
    noisesigma = args.noisesigma
    scale = args.scale; y  = args.y; F = args.F
    bscale = args.bscale
    Dx = args.Dx; Dy = args.Dy
    Db = args.Db

    logp = 0.0

    res = cache.residual
    Dxprop = cache.Dxprop
    Dyprop = cache.Dyprop
    Dbprop = cache.Dbprop
    Fxprop = cache.Fxprop
    G = cache.gradiprop


    #Fxprop .= F*x
    mul!(Fxprop,F,x)
    storesubst!(res,Fxprop,y)
    #res .= Fxprop - y  
    #Dxprop .= Dx*x
    mul!(Dxprop,Dx,x)
    #Dyprop .= Dy*x
    mul!(Dyprop,Dy,x)
    #Dbprop .= Db*x
    mul!(Dbprop,Db,x)

    mul!(G,F',-((res)/noisesigma.^2))

    if both
        logp =  -0.5/noisesigma^2*dotitself(res) -scale*dotitself(Dxprop)  -scale*dotitself(Dyprop)  -bscale*dotitself(Dbprop)
    end

    mul!(G,-2*scale*Dx',Dxprop,1,1)
    mul!(G,-2*scale*Dy',Dyprop,1,1)
    mul!(G,-2*bscale*Db',Dbprop,1,1)
    #println(maximum(G))

    return logp, G 
end

function  logpisodiff_cauchy(x,args,cache)
    noisesigma = args.noisesigma
    scale = args.scale; y  = args.y; F = args.F
    bscale = args.bscale
    Dx = args.Dx; Dy = args.Dy
    Db = args.Db

    res = cache.residual
    Dxprop = cache.Dxprop
    Dyprop = cache.Dyprop
    Dbprop = cache.Dbprop
    Fxprop = cache.Fxprop

    mul!(Fxprop,F,x)
    #res .= Fxprop - y  
    storesubst!(res,Fxprop,y)
    #Dxprop .= Dx*x
    mul!(Dxprop,Dx,x)
    #Dyprop .= Dy*x
    mul!(Dyprop,Dy,x)
    #Dbprop .= Db*x
    mul!(Dbprop,Db,x)

    return  -0.5/noisesigma^2*dotitself(res) -3/2*sum(log.(scale^2 .+ Dxprop.^2 + Dyprop.^2)) - sum(log.(bscale^2 .+ Dbprop.^2))

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

function regmatrices_second(dim)
    reg1d = spdiagm(Pair(0,2*ones(dim))) + spdiagm(Pair(1,-1*ones(dim-1))) + spdiagm(Pair(-1,-1*ones(dim-1))) #;reg1d[dim,dim] = 0
    #reg1d = reg1d[1:dim-1,:]
    iden = I(dim)
    regx = kron(reg1d,iden)
    regy = kron(iden,reg1d)

    rmxix = sum(abs.(regx), dims=2) .< 4
    rmyix = sum(abs.(regy), dims=2) .< 4
    boundary = ((rmxix + rmyix)[:]) .!= 0
    q = findall(boundary .== 1)
    regx = regx[setdiff(1:dim^2,q), :] 
    regy = regy[setdiff(1:dim^2,q), :] 

    #q = findall(boundary .== 1)
    #bmatrix = zeros(dim*dim,dim*dim)
    #for i in q
    #     bmatrix[i,i] = 1
    #end

    h2 = sparse(zeros(2,N))
    h2[1,1] = 1; h2[2,N] = 1
    h1 = hcat([sparse(zeros(dim-2)),I(dim-2),sparse(zeros(dim-2))]...)
    bmatrix = [ kron(h2,I(dim)); kron(h1,h2) ]

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


@inbounds  function measurementmatrix2d(X,Y,Xt,Yt,kernel;constant=10.0)
    dy = abs(Y[2,1]-Y[1,1])
    dx = abs(X[1,2]-X[1,1])
    d = Float64(dy*dx)
    N = length(X);
    Nt = length(Xt)
    #F = zeros(Nt,N)

    minval = 0.0001*d*kernel(0,0,0,0,constant=constant)
    Nth = Threads.nthreads()
    rows = Vector{Vector{Int64}}(undef, Nth)
    cols = Vector{Vector{Int64}}(undef, Nth)
    vals = Vector{Vector{Float64}}(undef, Nth)
    for p = 1:Nth
        rows[p] = []
        cols[p] = []
        vals[p] = []
    end

    Threads.@threads  for i = 1:N
        for j = 1:Nt
            xi = div(i-1,N)+1; 
            yi = rem(i,N)+1
            xj = div(j-1,Nt)+1; 
            yj = rem(j,Nt)+1
            val = d*kernel(X[i],Xt[j],Y[i],Yt[j],constant=constant);
            if (val >= minval)
                #F[j,i] = val
                push!(rows[Threads.threadid()],j)
                push!(cols[Threads.threadid()],i)
                push!(vals[Threads.threadid()],val)
            end
            
            #F[i,j] = d*kernel(X[xi],X[xj],Y[yi],Y[yj],constant=constant);
            #F[j,i] = F[i,j]
        end
    end
    rows = vcat(rows...)
    cols = vcat(cols...)
    vals = vcat(vals...)
    F = sparse(rows,cols,vals,Nt,N)

    return F

end



function  logpisodiff(lf,ulf,x,args,cache)
    noisesigma = args.noisesigma
    #scale = args.scale; 
    y  = args.y; F = args.F
    #bscale = args.bscale
    Dx = args.Dx; Dy = args.Dy
    Db = args.Db

    res = cache.residual
    Dxprop = cache.Dxprop
    Dyprop = cache.Dyprop
    Dbprop = cache.Dbprop
    Fxprop = cache.Fxprop
    ld = cache.ld
    lb = cache.lb

    mul!(Fxprop,F,x)
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

    return   -0.5/noisesigma^2*dotitself(res) + sum(lb)   + sum(ld)   

end

function  logpisodiffgradi(lf,ulf,glf,gulf,x,args,cache;both=true)
    noisesigma = args.noisesigma
    y  = args.y; F = args.F
    Dx = args.Dx; Dy = args.Dy
    Db = args.Db

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

    #Fxprop .= F*x
    mul!(Fxprop,F,x)
    storesubst!(res,Fxprop,y)
    mul!(Dxprop,Dx,x)
    mul!(Dyprop,Dy,x)
    mul!(Dbprop,Db,x)

    #G .= F'*(-((res)./noisesigma.^2))
    mul!(G,F',-((res)/noisesigma.^2))

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




# lf(x) = logdensity(alpha,abs(x),ip;gamma0=7.0)
# ulf(x) = logdensityuni(alpha,abs(x),ip1d;gamma0=2.0)
# glf(x) = logdensityderi(alpha,x,ip;gamma0=7.0)
# gulf(x) = logdensityunideri(alpha,x,ip1d;gamma0=2.0)

#arg = (F=F,Db=Db,Dx=Dx,Dy=Dy,scale=7.0,bscale=2.0,y=y,noisesigma=1.0)
#cac = (Dbprop=Db*x0,Dyprop=Dy*x0,Dxprop=Dx*x0,Fxprop=F*x0,ld=similar(Dx*x0),lb=similar(Db*x0),gld=similar(Dx*x0), glb=similar(Db*x0), gradiprop = similar(x0), residual=similar(y))

linspace(x,y,z) = range(x,stop=y,length=z)
@inline function meshgrid(x,y) 
    grid_a = [i for i in x, j in y]
    grid_b = [j for i in x, j in y]

    return grid_a,grid_b
end

cw = 150.0
kernel(xi,xj,yi,yj;constant=cw) = constant/pi*exp(-constant*((xi-xj)^2 + (yi-yj)^2) )
#tf(x,y) =  15.0*exp.(-20*sqrt.((x .- 0.3).^2 .+ (y .- 0.3).^2)) + 10*((y-x) .< 0.7).*((y-x) .>= -0.7).*((-y-x) .<= 0.8).*((-y-x) .>= 0.4).*(-x .+ 0.1)  + 5*(-x+y .< 1).*(-x+y .>= 0.8).*(abs.(x) .<= 1).*(abs.(y) .<= 1) + ( 50*0.25 .- 50*sqrt.((x.-0.5).^2+(y.+0.6).^2)).*(sqrt.((x.-0.5).^2+(y.+0.6).^2) .<= 0.25);


noisevar = 0.05

Nbig = 333
dimbig = Nbig
N = 256
dim = N
Nmeas = 100
dimmeas = Nmeas

xsbig = -1+1/dimbig:2/dimbig:1-1/(dimbig)
ysbig = -1+1/dimbig:2/dimbig:1-1/(dimbig)

xs = -1+1/dim:2/dim:1-1/(dim)
ys = -1+1/dim:2/dim:1-1/(dim)
Y,X = meshgrid(xs,xs)

xibig = linspace(-1,1,dimbig)
Ybig,Xbig = meshgrid(xibig,xibig)

# ximeas = linspace(-1,1,dimmeas)
# Ymeas,Xmeas = meshgrid(ximeas,ximeas)

Random.seed!(1)

using QuasiMonteCarlo
axt = linspace(-1,1,dimmeas)
tps = QuasiMonteCarlo.sample(Nmeas^2,[axt[1],axt[1]],[axt[end],axt[end]],LatticeRuleSample())
Ymeas = tps[1,:]
Xmeas = tps[2,:]
# Pv = Vector{Vector{Float64}}(undef,N*N)
# for i in eachindex(X)
#     Pv[i] = zeros(2)
#     Pv[i][1] = rY[i]
#     Pv[i][2] = X[i]
# end
# Pix = mapslices(q-> findmin(t->norm(t-q),Pv)[2],tps,dims=1)
# Pix = collect(Set(Pix))[:]
# sort!(Pix)
# P = [[rY[i],X[i]] for i in Pix]

# Nallmeas = length(Pix)
# yt = map(t-> t[1],P)
# xt = map(t-> t[2],P)
# is = zeros(Nallmeas)
# for i in eachindex(P)
#     is[i] = interp_meas(P[i][1],P[i][2])
# end
# Ix = Pix

function maski(X,Y)
    @assert length(size(X)) == 2 && size(X)[1] == size(X)[2] == size(Y)[1] == size(Y)[2]
    M = BitMatrix(zeros(size(X)))

    for i in eachindex(X)
        logi = sqrt( (X[i]-0.3)^2 + (Y[i]-0.5)^2  ) < 0.2 || (sqrt( (X[i]-0.3)^2 + (Y[i]-0.1)^2  )) < 0.3 || (sqrt( (X[i]+0.3)^2 + (Y[i]+0.6)^2  )) < 0.3 || (sqrt( (1.4*X[i] + 0.9*Y[i]+0.5)^2 + (Y[i]-0.2)^2  )) < 0.3
        M[i] = logi
    end
    
    return M
end

Random.seed!(1)

Zbig = zeros(Nbig,Nbig)
MC  = spdematrix(xibig,0.8,1);
kp = 0.2*(MC\randn(Nbig*Nbig)*100 .+ 1);
gpart = maski(Xbig,Ybig)
Zbig[gpart] = Zbig[gpart] + reshape(kp,Nbig,Nbig)[gpart]




# Zbig = tf(Xbig,Ybig)
@time Fbig = measurementmatrix2d(Xbig,Ybig,Xmeas,Ymeas,kernel,constant=cw); # Theory matrix.
@time F = measurementmatrix2d(X,Y,Xmeas,Ymeas,kernel,constant=cw); # Theory matrix. 

xisim = linspace(-1,1,dimmeas)
Ysim,Xsim = meshgrid(xisim,xisim)
@time Fbigsim = measurementmatrix2d(Xbig,Ybig,Xsim,Ysim,kernel,constant=cw); # Theory matrix.
gtcsim = Fbigsim*Zbig[:]

zx = Fbig*Zbig[:]

#tricontourf(Xmeas,Ymeas,zx,256)

mebig = reshape(zx,(dimmeas,dimmeas))
y = mebig[:] + randn(Nmeas*Nmeas,)*sqrt(noisevar)

x0 = zeros(N*N)

macro bscale()
    return :(1.0)
end

Dx,Dy,Db = regmatrices_first(N);




tall0 = "deconvot2/muut.mat"
matwrite(tall0,Dict("y"=>y,"gt"=>Zbig,"gtcbig"=>gtcsim))




for sca ∈  [0.003, 0.01, 0.1]
    for stab ∈ [0.51, 0.8,  1.4, 1.7 , 1.99]

        # sca = 0.003
        # stab = 0.7


        arg = (F=F,Db=Db,Dx=Dx,Dy=Dy,scale=sca,bscale=@bscale(),y=y,noisesigma=sqrt(noisevar))
        cac = (Dbprop=Db*x0,Dyprop=Dy*x0,Dxprop=Dx*x0,Fxprop=F*x0,ld=similar(Dx*x0),lb=similar(Db*x0),gld=similar(Dx*x0), glb=similar(Db*x0), gradiprop = similar(x0), residual=similar(y))


        lf(x) = logdensity(stab,abs(x),ip;gamma0=sca)
        ulf(x) = logdensityuni(stab,abs(x),ip1d;gamma0=@bscale())
        glf(x) = logdensityderi(stab,x,ip;gamma0=sca)
        gulf(x) = logdensityunideri(stab,x,ip1d;gamma0=@bscale())



        # target1(x) = -logpisodiff_cauchy(x,arg,cac)
        # target1grad(x) = -logpisodiffgradi_cauchy(x,arg,cac)[2]
        # res = Optim.optimize(target1, target1grad, 0*randn(N*N,),Optim.LBFGS(), Optim.Options(allow_f_increases=true,show_trace=true,iterations=7); inplace = false)
        # MAP_diff1 = res.minimizer

        target2(x) = -logpisodiff(lf,ulf,x,arg,cac)
        target2grad(x) = -logpisodiffgradi( lf,ulf,glf,gulf,x,arg,cac)[2]
        res = Optim.optimize(target2, target2grad, copy(x0),Optim.LBFGS(), Optim.Options(g_tol = 5e-7,allow_f_increases=true,show_trace=true,iterations=3000); inplace = false)
        MAP = res.minimizer

        tall = "deconvot2/"*string(stab)*"-"*string(sca)*"-"*string(N)*".mat"
        matwrite(tall,Dict("MAP"=>MAP))


        end

end



