using SparseArrays
using LinearAlgebra
using StatsBase

function rearranger(Nx,Ny)
#function rearranger(Nx,Ny,S)
    lix = LinearIndices((1:Ny+2,1:Nx+2))    

    p = 1
    a = 1
    cols = Vector{Int64}()
    rows = Vector{Int64}()
    vals = Vector{Int64}()

    pcols = Vector{Int64}()
    prows = Vector{Int64}()
    pvals = Vector{Int64}()
    
    for j = 2:Nx+1
          for i = 2:Ny+1
            push!(rows,lix[i,j])
            push!(cols,a)
            push!(vals,1)
            a = a + 1

            push!(prows,lix[i,j])
            push!(pcols,p)
            push!(pvals,1)
            p = p + 1
        end
    end

    M = sparse(rows,cols,vals,(Nx+2)*(Ny+2),(Nx)*(Ny))

    a = 1
    cols = Vector{Int64}()
    rows = Vector{Int64}()
    vals = Vector{Int64}()
    c = 2
    for i = 2:Ny+1
        push!(rows,lix[i,c-1])
        push!(cols,a)
        push!(vals,1)
        a = a + 1

        push!(prows,lix[i,c-1])
        push!(pcols,p)
        push!(pvals,1)
        p = p + 1
    end

    Ml = sparse(rows,cols,vals,(Nx+2)*(Ny+2),(Ny))

    a = 1
    cols = Vector{Int64}()
    rows = Vector{Int64}()
    vals = Vector{Int64}()
    c = Nx+1
    for i = 2:Ny+1
        push!(rows,lix[i,c+1])
        push!(cols,a)
        push!(vals,1)
        a = a + 1

        push!(prows,lix[i,c+1])
        push!(pcols,p)
        push!(pvals,1)
        p = p + 1
    end
    
    Mr = sparse(rows,cols,vals,(Nx+2)*(Ny+2),(Ny))

    a = 1
    cols = Vector{Int64}()
    rows = Vector{Int64}()
    vals = Vector{Int64}()
    c = 2#,Ny+1
    for i = 2:Nx+1  
        push!(rows,lix[c-1,i])
        push!(cols,a)
        push!(vals,1)
        a = a + 1

        push!(prows,lix[c-1,i])
        push!(pcols,p)
        push!(pvals,1)
        p = p + 1
    end

    Mt = sparse(rows,cols,vals,(Nx+2)*(Ny+2),(Nx))

    a = 1
    cols = Vector{Int64}()
    rows = Vector{Int64}()
    vals = Vector{Int64}()
    c = Ny+1
    for i = 2:Nx+1  
        push!(rows,lix[c+1,i])
        push!(cols,a)
        push!(vals,1)
        a = a + 1

        push!(prows,lix[c+1,i])
        push!(pcols,p)
        push!(pvals,1)
        p = p + 1
    end

    Mb = sparse(rows,cols,vals,(Nx+2)*(Ny+2),(Nx))

    Nw = sparse([1],[1],[1],(Nx+2)*(Ny+2),1)
    Sw = sparse([Ny+2],[1],[1],(Nx+2)*(Ny+2),1)
    Ne = sparse([(Ny+2)*(Nx+2)-(Ny+1)],[1],[1],(Nx+2)*(Ny+2),1)
    Se = sparse([(Ny+2)*(Nx+2)],[1],[1],(Nx+2)*(Ny+2),1)

    Pnw = sparse([1],[1],[1],(Nx+2)*(Ny+2),(Nx+2)*(Ny+2))
    Psw = sparse([Ny+2],[1],[1],(Nx+2)*(Ny+2),(Nx+2)*(Ny+2))
    Pne = sparse([(Ny+2)*(Nx+2)-(Ny+1)],[1],[1],(Nx+2)*(Ny+2),(Nx+2)*(Ny+2))
    Pse = sparse([(Ny+2)*(Nx+2)],[1],[1],(Nx+2)*(Ny+2),(Nx+2)*(Ny+2))

    P = sparse(prows,pcols,pvals,(Nx+2)*(Ny+2),(Nx+2)*(Ny+2)) + Pnw + Psw + Pne + Pse

    return M,Ml,Mr,Mt,Mb,Nw,Sw,Ne,Se,P
end

function coefsbderi(data,b1,b2)
    N = length(data)
    cols = Vector{Int64}()
    rows = Vector{Int64}()
    vals = Vector{Float64}()

    for i = 2:N+1
        # S[i,i] = 2/3
        # S[i,i-1] = 1/6
        # S[i,i+1] = 1/6

        push!(rows,i)
        push!(cols,i)
        push!(vals,2/3)

        push!(rows,i)
        push!(cols,i-1)
        push!(vals,1/6)

        push!(rows,i)
        push!(cols,i+1)
        push!(vals,1/6)
    end

    # S[1,1] = -1/2
    # S[1,3] = 1/2

    push!(rows,1)
    push!(cols,1)
    push!(vals,-1/2)

    push!(rows,1)
    push!(cols,3)
    push!(vals,1/2)

    # S[N+2,N] = -1/2
    # S[N+2,N+2] = 1/2

    push!(rows,N+2)
    push!(cols,N)
    push!(vals,-1/2)

    push!(rows,N+2)
    push!(cols,N+2)
    push!(vals,1/2)

    v = [b1;data;b2]

    S = sparse(rows,cols,vals,N+2,N+2)

    coefs = S\v
    return coefs

end



function coefsbderi2(data,leftdx,bottomdy,rightdx,topdy)
    Nx = size(data)[2]
    Ny = size(data)[1]
    # cols = Vector{Int64}()
    # rows = Vector{Int64}()
    # vals = Vector{Float64}()

    lix = LinearIndices((1:Ny+2,1:Nx+2))

    #rox = 1

    Nth = Threads.nthreads()
    tvals = Vector{Vector{Float64}}(undef,Nth)
    tcols = Vector{Vector{Int64}}(undef,Nth)
    trows = Vector{Vector{Int64}}(undef,Nth)
    troxs = Vector{Vector{Int64}}(undef,Nth)

    for t = 1:Nth
        tvals[t] = Float64[]
        tcols[t] = Int64[]
        trows[t] = Int64[]
        troxs[t] = [1,]
    end

    gcols = zeros(Int64,9*Nx*Ny)
    grows = zeros(Int64,9*Nx*Ny)
    gvals = zeros(Float64,9*Nx*Ny)

    #R = Threads.Atomic{Int64}(1)
    #G = Threads.Atomic{Int64}(1)

   @time for j = 2:Nx+1
       
        # t = Threads.threadid()
        # å = roxs[Threads.threadid()]
          for i = 2:Ny+1
            # S[i,i] = 2/3
            # S[i,i-1] = 1/6
            # S[i,i+1] = 1/6
            
          

            # lix = lx# lixs[t]

            node = lix[i,j]
            #rox = node
            node_0p = lix[i,j+1]
            node_p0 = lix[i+1,j]
            node_pp = lix[i+1,j+1]
            node_0m = lix[i,j-1]
            node_m0 = lix[i-1,j]
            node_mm = lix[i-1,j-1]
            node_mp = lix[i-1,j+1]
            node_pm = lix[i+1,j-1]

            # t = Threads.threadid()
            
            # å = roxs[t]
            # rox = å[1] 
            # #rox = Threads.atomic_add!(R, 1)
            
            
            # push!(trows[t],rox)
            # push!(tcols[t],node)
            # push!(tvals[t],4/9)

            # push!(trows[t],rox)
            # push!(tcols[t],node_mm)
            # push!(tvals[t],1/36)

            # push!(trows[t],rox)
            # push!(tcols[t],node_m0)
            # push!(tvals[t],1/9)

            # push!(trows[t],rox)
            # push!(tcols[t],node_0m)
            # push!(tvals[t],1/9)

            # push!(trows[t],rox)
            # push!(tcols[t],node_pm)
            # push!(tvals[t],1/36)

            # push!(trows[t],rox)
            # push!(tcols[t],node_mp)
            # push!(tvals[t],1/36)

            # push!(trows[t],rox)
            # push!(tcols[t],node_0p)
            # push!(tvals[t],1/9)

            # push!(trows[t],rox)
            # push!(tcols[t],node_p0)
            # push!(tvals[t],1/9)

            # push!(trows[t],rox)
            # push!(tcols[t],node_pp)
            # push!(tvals[t],1/36)

            # å[1] = å[1]  + 1


            H = 9*( (j-2)*Ny + (i-1) -1)+1
            rox = (j-2)*Ny + (i-1) 

            grows[H] = rox
            gcols[H] = node
            gvals[H] = 4/9

            H = H + 1

            grows[H] = rox
            gcols[H] = node_mm
            gvals[H] = 1/36

            H = H + 1

            grows[H] = rox
            gcols[H] = node_m0
            gvals[H] = 1/9

            H = H + 1

            grows[H] = rox
            gcols[H] = node_0m
            gvals[H] = 1/9

            H = H + 1

            grows[H] = rox
            gcols[H] = node_pm
            gvals[H] = 1/36

            H = H + 1

            grows[H] = rox
            gcols[H] = node_mp
            gvals[H] = 1/36

            H = H + 1

            grows[H] = rox
            gcols[H] = node_0p
            gvals[H] = 1/9

            H = H + 1

            grows[H] = rox
            gcols[H] = node_p0
            gvals[H] = 1/9

            H = H + 1

            grows[H] = rox
            gcols[H] = node_pp
            gvals[H] = 1/36


        end
    end

    # icols = copy( vcat(tcols...))
    # irows = copy( vcat(trows...))
    # ivals = copy( vcat(tvals...))
    icols = gcols
    irows = grows
    ivals = gvals
    # q = [(irows[i],icols[i]) for i in eachindex(icols)]
    # q2 = Set()
    # for  i in eachindex(icols)
    #     p = (irows[i],icols[i])
    #     if p ∉ q2      
    #         push!(q2,p)
    #     else
    #         println(p)
    #     end
    # end
    # println(length(q),",",length(q2))
    # plot(irows)
    S = sparse(irows,icols,ivals,Nx*Ny,(Nx+2)*(Ny+2))
    

    # S = sparse(rows[1],cols[1],vals[1],roxs[1][1]-1,(Nx+2)*(Ny+2))

    # for t = 2:Nth
    #     #display(maximum(rows[t))
    #     #display(roxs[t])
    #     #display()
    #     U = sparse(rows[t],cols[t],vals[t],roxs[t][1]-1,(Nx+2)*(Ny+2))
    #     S = [S;U]
    # end

    #display(S)
    # println(countmap(irows))
    # println(countmap(icols))
    # error("")
    #display(ivals)


    rcols = Vector{Int64}()
    rrows = Vector{Int64}()
    rvals = Vector{Float64}()

    lcols = Vector{Int64}()
    lrows = Vector{Int64}()
    lvals = Vector{Float64}()


    qox = [[1,], [1,]]

    for c = [2,Nx+1] 
         for i = 2:Ny+1

            if c == 2
                rox = qox[1]
                rows = lrows
                cols = lcols
                vals = lvals

            else
                rox = qox[2]
                rows = rrows
                cols = rcols
                vals = rvals
            end
              
            node = lix[i,c]
            r = rox[1]
            node_mp = lix[i-1,c+1]
            node_0p = lix[i,c+1]
            node_pp = lix[i+1,c+1]
            node_mm = lix[i-1,c-1]
            node_0m = lix[i,c-1]
            node_pm = lix[i+1,c-1]
            
            
            push!(rows,r)
            push!(cols,node_mp)
            push!(vals,1/12)

            push!(rows,r)
            push!(cols,node_0p)
            push!(vals,1/3)

            push!(rows,r)
            push!(cols,node_pp)
            push!(vals,1/12)

            push!(rows,r)
            push!(cols,node_mm)
            push!(vals,-1/12)

            push!(rows,r)
            push!(cols,node_0m)
            push!(vals,-1/3)

            push!(rows,r)
            push!(cols,node_pm)
            push!(vals,-1/12)

            rox[1] = rox[1] + 1

        end

    end
    
    # for i = 2:Ny+1

    #     c = Nx+1

    #     node = lix[i,c]
    #     r = rox
    #     node_pm = lix[i+1,c-1]
    #     node_p0 = lix[i+1,c]
    #     node_pp = lix[i+1,c+1]
    #     node_mm = lix[i-1,c-1]
    #     node_m0 = lix[i-1,c]
    #     node_mp = lix[i-1,c+1]
             
        

    #     push!(rows,r)
    #     push!(cols,node_pm)
    #     push!(vals,1/12)

    #     push!(rows,r)
    #     push!(cols,node_p0)
    #     push!(vals,1/3)

    #     push!(rows,r)
    #     push!(cols,node_pp)
    #     push!(vals,1/12)

    #     push!(rows,r)
    #     push!(cols,node_mm)
    #     push!(vals,-1/12)

    #     push!(rows,r)
    #     push!(cols,node_m0)
    #     push!(vals,-1/3)

    #     push!(rows,r)
    #     push!(cols,node_mp)
    #     push!(vals,-1/12)

    #     rox = rox + 1

    # end

    tcols = Vector{Int64}()
    trows = Vector{Int64}()
    tvals = Vector{Float64}()

    bcols = Vector{Int64}()
    brows = Vector{Int64}()
    bvals = Vector{Float64}()


    pox = [[1,], [1,]]
   
    for c = [2,Ny+1]
        for i = 2:Nx+1       
            
            if c == 2
                rox = pox[1]
                rows = trows
                cols = tcols
                vals = tvals

            else
                rox = pox[2]
                rows = brows
                cols = bcols
                vals = bvals
            end
              
            node = lix[c,i]
            r = rox[1]
            node_pm = lix[c+1,i-1]
            node_p0 = lix[c+1,i]
            node_pp = lix[c+1,i+1]
            node_mm = lix[c-1,i-1]
            node_m0 = lix[c-1,i]
            node_mp = lix[c-1,i+1]         
            

            push!(rows,r)
            push!(cols,node_pm)
            push!(vals,1/12)

            push!(rows,r)
            push!(cols,node_p0)
            push!(vals,1/3)

            push!(rows,r)
            push!(cols,node_pp)
            push!(vals,1/12)

            push!(rows,r)
            push!(cols,node_mm)
            push!(vals,-1/12)

            push!(rows,r)
            push!(cols,node_m0)
            push!(vals,-1/3)

            push!(rows,r)
            push!(cols,node_mp)
            push!(vals,-1/12)

            rox[1] = rox[1] + 1

        end
    end

     
    # for i = 2:Nx+1

    #     c = Ny+1

    #     node = lix[c,i+1]
    #     r = rox
    #     node_mp = lix[c-1,i+1]
    #     node_0p = lix[c,i+1]
    #     node_pp = lix[c+1,i+1]
    #     node_mm = lix[c-1,i-1]
    #     node_0m = lix[c,i-1]
    #     node_pm = lix[c+1,i-1]         
        

    #     push!(rows,r)
    #     push!(cols,node_mp)
    #     push!(vals,1/12)

    #     push!(rows,r)
    #     push!(cols,node_0p)
    #     push!(vals,1/3)

    #     push!(rows,r)
    #     push!(cols,node_pp)
    #     push!(vals,1/12)

    #     push!(rows,r)
    #     push!(cols,node_mm)
    #     push!(vals,-1/12)

    #     push!(rows,r)
    #     push!(cols,node_0m)
    #     push!(vals,-1/3)

    #     push!(rows,r)
    #     push!(cols,node_pm)
    #     push!(vals,-1/12)

    #     rox = rox + 1

    # end

    #=
    for i = [2,Nx+1] #2:Nx+1

        c = 2# Ny+1

        node = lix[c,i]
        r = rox
        node_mp = lix[c-1,i+1]
        node_m0 = lix[c-1,i]
        node_0p = lix[c,i+1]
        node_p0 = lix[c+1,i]
        node_pp = lix[c+1,i+1]
        node_mm = lix[c-1,i-1]
        node_mp = lix[c-1,i+1]
        node_0m = lix[c,i-1]
        node_pm = lix[c+1,i-1]     
        

        push!(rows,r)
        push!(cols,node_mm)
        push!(vals,1/6)
        
        push!(rows,r)
        push!(cols,node_m0)
        push!(vals,2/3)

        push!(rows,r)
        push!(cols,node_mp)
        push!(vals,1/6)

        push!(rows,r)
        push!(cols,node_pm)
        push!(vals,1/6)
  
        push!(rows,r)
        push!(cols,node_p0)
        push!(vals,2/3)

        push!(rows,r)
        push!(cols,node_pp)
        push!(vals,1/6)

        push!(rows,r)
        push!(cols,node_0m)
        push!(vals,-1/3)

        push!(rows,r)
        push!(cols,node_0p)
        push!(vals,-1/3)

        push!(rows,r)
        push!(cols,node)
        push!(vals,-4/3)

        rox = rox + 1

    end

    
    for i = [2,Ny+1] #2:Nx+1

        c =  Nx+1

        node = lix[i,c]
        r = rox
        node_pm = lix[i+1,c-1]
        node_0m = lix[i,c-1]
        node_p0 = lix[i+1,c]
        node_0p = lix[i,c+1]
        node_pp = lix[i+1,c+1]
        node_mm = lix[i-1,c-1]
        node_pm = lix[i+1,c-1]
        node_m0 = lix[i-1,c]
        node_mp = lix[i-1,c+1]     
        

        push!(rows,r)
        push!(cols,node_mm)
        push!(vals,1/6)
        
        push!(rows,r)
        push!(cols,node_mp)
        push!(vals,1/6)

        push!(rows,r)
        push!(cols,node_0m)
        push!(vals,2/3)

        push!(rows,r)
        push!(cols,node_0p)
        push!(vals,2/3)
  
        push!(rows,r)
        push!(cols,node_pm)
        push!(vals,1/6)

        push!(rows,r)
        push!(cols,node_pp)
        push!(vals,1/6)

        push!(rows,r)
        push!(cols,node_m0)
        push!(vals,-1/3)

        push!(rows,r)
        push!(cols,node)
        push!(vals,-4/3)

        push!(rows,r)
        push!(cols,node_p0)
        push!(vals,-1/3)

        rox = rox + 1

    end
    =#

    nwcols = Vector{Int64}()
    nwrows = Vector{Int64}()
    nwvals = Vector{Float64}()

    rox = 1

    push!(nwrows,rox)
    push!(nwcols,lix[1,1])
    push!(nwvals,1/2)

    push!(nwrows,rox)
    push!(nwcols,lix[2,2])
    push!(nwvals,-1)

    push!(nwrows,rox)
    push!(nwcols,lix[3,3])
    push!(nwvals,1/2)

    Bnw = sparse(nwrows,nwcols,nwvals,1,(Nx+2)*(Ny+2))

    necols = Vector{Int64}()
    nerows = Vector{Int64}()
    nevals = Vector{Float64}()

    push!(nerows,rox)
    push!(necols,lix[3,end-2])
    push!(nevals,1/2)

    push!(nerows,rox)
    push!(necols,lix[2,end-1])
    push!(nevals,-1)

    push!(nerows,rox)
    push!(necols,lix[1,end])
    push!(nevals,1/2)

    Bne = sparse(nerows,necols,nevals,1,(Nx+2)*(Ny+2))

    swcols = Vector{Int64}()
    swrows = Vector{Int64}()
    swvals = Vector{Float64}()

    push!(swrows,rox)
    push!(swcols,lix[end,1])
    push!(swvals,1/2)

    push!(swrows,rox)
    push!(swcols,lix[end-1,2])
    push!(swvals,-1)

    push!(swrows,rox)
    push!(swcols,lix[end-2,3])
    push!(swvals,1/2)

    Bsw = sparse(swrows,swcols,swvals,1,(Nx+2)*(Ny+2))

    secols = Vector{Int64}()
    serows = Vector{Int64}()
    sevals = Vector{Float64}()


    push!(serows,rox)
    push!(secols,lix[end,end])
    push!(sevals,1/2)

    push!(serows,rox)
    push!(secols,lix[end-1,end-1])
    push!(sevals,-1)

    push!(serows,rox)
    push!(secols,lix[end-2,end-2])
    push!(sevals,1/2)

    Bse = sparse(serows,secols,sevals,1,(Nx+2)*(Ny+2))

    #Nr = rox 

    

    Bl = sparse(lrows,lcols,lvals,qox[1][1]-1,(Nx+2)*(Ny+2))
    Br = sparse(rrows,rcols,rvals,qox[2][1]-1,(Nx+2)*(Ny+2))
    Bt = sparse(trows,tcols,tvals,pox[1][1]-1,(Nx+2)*(Ny+2))
    Bb = sparse(brows,bcols,bvals,pox[2][1]-1,(Nx+2)*(Ny+2))

    #B = sparse(rows,cols,vals,Nr,(Nx+2)*(Ny+2))

    #Ba = sparse(rows,cols,vals,4,(Nx+2)*(Ny+2))

    M,Ml,Mr,Mt,Mb,Nw,Sw,Ne,Se,P = rearranger(Nx,Ny)

    #K = [S;B]
    
    ## Permutation reduces the condition number a lot when the system is large. It is also crucial, otherwise
    ## numerical precision issues emerge!

    Q = M*S+Ml*Bl+Mr*Br+Mt*Bt+Mb*Bb+Nw*Bnw+Sw*Bsw+Ne*Bne+Se*Bse
    K = [S;Bl;Br;Bt;Bb;Bnw;Bsw;Bne;Bse]
    v = [data[:];leftdx;rightdx;topdy;bottomdy;zeros(4)]
    #f = P*v

  
    vv = P*v

    @time coefs = Q\vv
    return Q,vv,coefs

end