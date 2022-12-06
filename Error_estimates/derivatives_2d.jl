using Statistics, SpecialFunctions, MAT, HDF5, CairoMakie
CairoMakie.activate!(type="svg")

M_2D = matread("hila_2d.mat")["M"];

# The grid's resolution Δ_M is parametrized by ϰ, a 
# positive integer. The code below is mathematically
# sensible for ϰ ∈ {1,2,5,10}.

ϰ = 2;

M = M_2D[1:ϰ:1451,1:ϰ:30001];

Δ_M = ϰ*1e-3;
A = 0.5:Δ_M:1.9;
R = 0.0:Δ_M:30.0;

# Load precomputed integrals

A_coarse = h5read("integrals_2d.h5","A");
R0 = h5read("integrals_2d.h5","R0");
I∞ = h5read("integrals_2d.h5","I∞");
U = h5read("integrals_2d.h5","U");
osc = h5read("integrals_2d.h5","osc");

i_coarse(i) = min( findlast(isone, A_coarse .<= A[i]), length(A_coarse)-1 )

function series0(i,j,n,ℓ₁,ℓ₂)
    if n < 0
        return 0.0
    end
    
    Σ = zeros(n+1)
    α₋ = A[i]
    α₊ = A[i+1]
    
    Π = [ prod([ 2*k+isodd(ℓ₁)+i for i ∈ 1:ℓ₁ ]) for k ∈ 0:n ]
    z = [ 2*k+2+2*ceil(Int,ℓ₁/2) for k ∈ 0:n ]
    w = z./α₊
    
    # We will compute the series for ℓ₂ ∈ {0,1,2}. For ℓ₂ ≥ 3 the computations get
    # unnecessarily wieldy so we will settle for the uniform bound.
    
    if ℓ₂ == 0
        for k ∈ 0:n
            # See the case ℓ₂ == 1 below for the meaning of the following
            if w[k+1] >= 1.5
                g₊ = gamma(z[k+1]/α₋)/α₋
            elseif w[k+1] <= 1.4
                g₊ = gamma(z[k+1]/α₊)/α₋
            else
                g₊ = gamma(1.55)/α₋
            end
            
            Σ[k+1] = Π[k+1]*g₊*R[j+1]^(2*k + isodd(ℓ₁)) / ( 2^(k+ceil(Int,ℓ₁/2))*factorial(k+ceil(Int,ℓ₁/2)) )^2
        end
        return sum(Σ)/(2*π)
        
    elseif ℓ₂ == 1
        # From SymPy we have diff(gamma(z/α)/α, α) = -(z*polygamma(0, z/α) + α)*gamma(z/α)/α^3.
        # We note that gamma(x) and polygamma(0,x) are positive and increasing functions for x ≥ 1.5,
        # and that their absolute values are decreasing for 0.5 ≤ x ≤ 1.46
        for k ∈ 0:n
            if w[k+1] >= 1.5
                # If z[k+1]/A[i+1] ≥ 1.5, then the same holds with A[i], by monotonicity
                d₁ = gamma(z[k+1]/α₋)
                d₂ = polygamma(0, z[k+1]/α₋)
            elseif w[k+1] <= 1.4
                # If z[k+1]/A[i+1] ≤ 1.4, then z[k+1]/A[i] ≤ 1.46 by the grid's properties
                d₁ = gamma(z[k+1]/α₊)
                d₂ = abs(polygamma(0, z[k+1]/α₊))
            else
                d₁ = gamma(1.55)
                d₂ = abs(polygamma(0,1.55))
            end
            d₊ = (z[k+1]*d₂ + α₋)*d₁/α₋^3
            Σ[k+1] = Π[k+1]*d₊*R[j+1]^(2*k + isodd(ℓ₁)) / ( 2^(k+ceil(Int,ℓ₁/2))*factorial(k+ceil(Int,ℓ₁/2)) )^2
        end
        return sum(Σ)/(2*π)
        
    elseif ℓ₂ == 2
        # We have diff(gamma(z/α)/α, α, 2) = (z^2*polygamma(0, z/α)^2 +
        #     z^2*polygamma(1, z/α) + 4*z*α*polygamma(0, z/α) + 2*α^2)*gamma(z/α)/α^5.
        # The function polygamma(1,⋅) is positive and decreasing on the entire positive
        # real line; see Theorem 1.2.5 in https://doi.org/10.1017/CBO9781107325937
        for k ∈ 0:n
            if w[k+1] >= 1.5
                d₁ = gamma(z[k+1]/α₋)
                d₂ = polygamma(0, z[k+1]/α₋)
            elseif w[k+1] <= 1.4
                d₁ = gamma(z[k+1]/α₊)
                d₂ = abs(polygamma(0, z[k+1]/α₊))
            else
                d₁ = gamma(1.55)
                d₂ = abs(polygamma(0,1.55))
            end
            d₃ = polygamma(1,z[k+1]/α₊)
            d₊ = (z[k+1]^2*d₂^2 + z[k+1]^2*d₃ + 4*z[k+1]*α₋*d₂ + 2*α₋^2)*d₁/α₋^5
            Σ[k+1] = Π[k+1]*d₊*R[j+1]^(2*k + isodd(ℓ₁)) / ( 2^(k+ceil(Int,ℓ₁/2))*factorial(k+ceil(Int,ℓ₁/2)) )^2
        end
        return sum(Σ)/(2*π)
        
    else
        return 0.0
    end
end

function remainder0(i,j,n,ℓ₁,ℓ₂)
    return R[j+1]^(2*n+2+isodd(ℓ₁))*R0[i_coarse(i),ℓ₁+1,ℓ₂+1,n+1]/(2*π*A[i]^(ℓ₂+1)*factorial(2*n+2+isodd(ℓ₁)))
end

function series∞(i,j,n,ℓ₁,ℓ₂)
    if n <= 0
        return 0.0
    end
    
    Σ = zeros(n)
    α₋ = A[i]
    α₊ = A[i+1]
    logr₊ = maximum(abs.([log(R[j]),log(R[j+1])]))
    
    w = [ k*α₋/2 + 1 for k ∈ 1:n ]
    z = [ k*α₊/2 + 1 for k ∈ 1:n ]
    
    # Here we consider the cases with max(ℓ₁,2) + ℓ₂ ≤ 2, settling for the
    # case n = 0 (i.e. empty sum) for the higher orders
    
    if ℓ₂ == 0
        for k ∈ 1:n
            Π = prod([k*α₊ + i for i ∈ 2:(ℓ₁+1)])
            ω = k*π/2
            s₊ = abs(sin(ω*α₋)) + Δ_M*ω   # Upper bound for sin(α*ω)
            #c₊ = abs(cos(ω*α₋)) + Δ_M*ω   # Upper bound for cos(α*ω)
            p₋ = R[j] >= 1.0 ? R[j]^(k*α₋ + ℓ₁ + 2) : R[j]^(k*α₊ + ℓ₁ + 2)
            Σ[k] = 2^(k*α₊)*Π*gamma((k*α₊+2)/2)^2*s₊/(p₋*factorial(k))
        end
        return sum(Σ)/π^2
    
    elseif (ℓ₂ == 1 && ℓ₁ == 1)
        # We have
        # diff(2^(k*α) * (k*α+2) * gamma((k*α+2)/2)^2 * sin(ω*α) / r^(k*α+3) , α)
        #    = 2^(k*α)*(-k^2*α*log(r)*sin(α*ω) + k^2*α*sin(α*ω)*polygamma(0, k*α/2 + 1) +
        #        k^2*α*log(2)*sin(α*ω) + k*α*ω*cos(α*ω) - 2*k*log(r)*sin(α*ω) + 
        #        2*k*sin(α*ω)*polygamma(0, k*α/2 + 1) + k*sin(α*ω) + 2*k*log(2)*sin(α*ω) +
        #        2*ω*cos(α*ω))*gamma(k*α/2 + 1)^2/r^(k*α + 3) +
        for k ∈ 1:n
            if w[k] >= 1.5
                d₁ = gamma(z[k])
                d₂ = polygamma(0,z[k])
            elseif w[k] <= 1.4
                d₁ = gamma(w[k])
                d₂ = abs(polygamma(0,w[k]))
            else
                d₁ = gamma(1.55)
                d₂ = abs(polygamma(0,1.55))
            end
            ω = k*π/2
            s₊ = abs(sin(ω*α₋)) + Δ_M*ω   # Upper bound for sin(α*ω)
            c₊ = abs(cos(ω*α₋)) + Δ_M*ω   # Upper bound for cos(α*ω)
            p₋ = R[j] >= 1.0 ? R[j]^(k*α₋ + 3) : R[j]^(k*α₊ + 3)
            d₊ = 2^(k*α₊)*(k^2*α₊*logr₊*s₊ + k^2*α₊*s₊*d₂ +
                k^2*α₊*log(2)*s₊ + k*α₊*ω*c₊ + 2*k*logr₊*s₊ +
                2*k*s₊*d₂ + k*s₊ + 2*k*log(2)*s₊ +
                2*ω*c₊)*d₁^2/p₋
            Σ[k] = d₊/factorial(k)
        end
        return sum(Σ)/π^2
        
    elseif (ℓ₂ == 1 && ℓ₁ == 0)
        # diff(2^(k*α) * gamma((k*α+2)/2)^2 * sin(ω*α) / r^(k*α+2) , α)
        #    = 2^(k*α)*(-k*log(r)*sin(α*ω) + k*sin(α*ω)*polygamma(0, k*α/2 + 1) +
        #         k*log(2)*sin(α*ω) + ω*cos(α*ω))*gamma(k*α/2 + 1)^2/r^(k*α + 2)
        for k ∈ 1:n
            if w[k] >= 1.5
                d₁ = gamma(z[k])
                d₂ = polygamma(0,z[k])
            elseif w[k] <= 1.4
                d₁ = gamma(w[k])
                d₂ = abs(polygamma(0,w[k]))
            else
                d₁ = gamma(1.55)
                d₂ = abs(polygamma(0,1.55))
            end
            ω = k*π/2
            s₊ = abs(sin(ω*α₋)) + Δ_M*ω   # Upper bound for sin(α*ω)
            c₊ = abs(cos(ω*α₋)) + Δ_M*ω   # Upper bound for cos(α*ω)
            p₋ = R[j] >= 1.0 ? R[j]^(k*α₋ + 2) : R[j]^(k*α₊ + 2)
            d₊ = 2^(k*α₊)*(k*logr₊*s₊ + k*s₊*d₂ + k*log(2)*s₊ + ω*c₊)*d₁^2/p₋
            Σ[k] = d₊/factorial(k)
        end
        return sum(Σ)/π^2
        
    elseif (ℓ₂ == 2 && ℓ₁ == 0)
        # diff(2^(k*α) * gamma((k*α+2)/2)^2 * sin(ω*α) / r^(k*α+2) , α, 2)
        #   = 2^(k*α - 1)*(2*k^2*log(r)^2*sin(α*ω) - 4*k^2*log(r)*sin(α*ω)*polygamma(0, k*α/2 + 1) -
        #        4*k^2*log(2)*log(r)*sin(α*ω) + 2*k^2*sin(α*ω)*polygamma(0, k*α/2 + 1)^2 +
        #        4*k^2*log(2)*sin(α*ω)*polygamma(0, k*α/2 + 1) + k^2*sin(α*ω)*polygamma(1, k*α/2 + 1) +
        #        2*k^2*log(2)^2*sin(α*ω) - 4*k*ω*log(r)*cos(α*ω) + 4*k*ω*cos(α*ω)*polygamma(0, k*α/2 + 1) +
        #        4*k*ω*log(2)*cos(α*ω) - 2*ω^2*sin(α*ω))*gamma(k*α/2 + 1)^2/r^(k*α + 2)
        for k ∈ 1:n
            if w[k] >= 1.5
                d₁ = gamma(z[k])
                d₂ = polygamma(0,z[k])
            elseif w[k] <= 1.4
                d₁ = gamma(w[k])
                d₂ = abs(polygamma(0,w[k]))
            else
                d₁ = gamma(1.55)
                d₂ = abs(polygamma(0,1.55))
            end
            d₃ = polygamma(1,w[k])
            ω = k*π/2
            s₊ = abs(sin(ω*α₋)) + Δ_M*ω   # Upper bound for sin(α*ω)
            c₊ = abs(cos(ω*α₋)) + Δ_M*ω   # Upper bound for cos(α*ω)
            p₋ = R[j] >= 1.0 ? R[j]^(k*α₋ + 2) : R[j]^(k*α₊ + 2)
            d₊ = 2^(k*α₊ - 1)*(2*k^2*logr₊^2*s₊ - 4*k^2*logr₊*s₊*d₂ +
                4*k^2*log(2)*logr₊*s₊ + 2*k^2*s₊*d₂ +
                4*k^2*log(2)*s₊*d₂ + k^2*s₊*d₃ +
                2*k^2*log(2)^2*s₊ + 4*k*ω*logr₊*c₊ + 4*k*ω*c₊*d₂ +
                4*k*ω*log(2)*c₊ + 2*ω^2*s₊)*d₁^2/p₋
            Σ[k] = d₊/factorial(k)
        end
        return sum(Σ)/π^2

    else
        return 0.0
    end
end

# Auxiliary functions for the estimates below.
π_(α) = π/(2*max(1,α))

function remainder∞(i,j,n,ℓ₁,ℓ₂)
    α₋ = A[i]
    α₊ = A[i+1]
    logr₊ = maximum(abs.([log(R[j]),log(R[j+1])]))
    
    if ℓ₂ == 0
        p₋ = R[j] >= 1.0 ? R[j]^((n+1)*α₋ + ℓ₁ + 2) : R[j]^((n+1)*α₊ + ℓ₁ + 2)
        return I∞[i_coarse(i),ℓ₁+1,1,n+1]/(2*π*factorial(n+1)*p₋)
        
    elseif (ℓ₂ == 1 && ℓ₁ <= 1)
        # Upper bounds for M_{n+1}(z) and its derivative for z with negative real part
        m = zeros(2)
        m[1] = 1/factorial(n+1)
        m[2] = 1/factorial(n+1) + (n+1)/factorial(n+2)
        
        p₋ = [ R[j] >= 1.0 ? R[j]^(k*α₋ + ℓ₁ + 2) : R[j]^(k*α₊ + ℓ₁ + 2) for k ∈ (n+1):(n+2) ]
        
        I_ = [
            2^(ℓ₂-1)*m[k-n]*p₋[k-n]^(-1)*(
                (logr₊ + π_(A[i+1]))^(ℓ₂) * I∞[i_coarse(i),ℓ₁+1,0+1,k] +
                I∞[i_coarse(i),ℓ₁+1,ℓ₂+1,k]
                ) for k ∈ (n+1):(n+2) ]
        return ((n+1)*I_[1] + I_[2])/(2*π)
        
    elseif (ℓ₂ == 2 && ℓ₁ == 0)
        # Upper bounds for M_{n+1}(z) and its two first derivatives
        m = zeros(3)
        m[1] = 1/factorial(n+1)
        m[2] = 1/factorial(n+1) + (n+1)/factorial(n+2)
        m[3] = 1/factorial(n+1) + 2*(n+1)/factorial(n+2) + (n+1)*(n+2)/factorial(n+3)
        
        p₋ = [ R[j] >= 1.0 ? R[j]^(k*α₋ + ℓ₁ + 2) : R[j]^(k*α₊ + ℓ₁ + 2) for k ∈ (n+1):(n+3) ]
        
        I_ = [
            2^(ℓ₂-1)*m[k-n]*p₋[k-n]^(-1)*(
                (logr₊ + π_(A[i+1]))^(ℓ₂) * I∞[i_coarse(i),ℓ₁+1,0+1,k] +
                I∞[i_coarse(i),ℓ₁+1,ℓ₂+1,k]
            ) for k ∈ (n+1):(n+3) ]
        return ((n+1)^2*I_[1] + (2*n+3)*I_[2] + I_[3])/(2*π)
        
    elseif (ℓ₂ == 1 && ℓ₁ >= 2)
        m = zeros(2)
        m[1] = 1
        m[2] = 3/2
        
        p₋ = [ R[j] >= 1.0 ? R[j]^(k*α₋ + ℓ₁ + 2) : R[j]^(k*α₊ + ℓ₁ + 2) for k ∈ 1:2 ]
        
        I_ = [
            2^(ℓ₂-1)*m[k]*p₋[k]^(-1)*(
                (logr₊ + π_(A[i+1]))^(ℓ₂) * I∞[i_coarse(i),ℓ₁+1,0+1,k] +
                I∞[i_coarse(i),ℓ₁+1,ℓ₂+1,k]
            ) for k ∈ 1:2 ]
        return (I_[1] + I_[2])/(2*π)
        
    elseif (ℓ₂ == 2 && ℓ₁ >= 1)
         # Upper bounds for M_1 and its derivatives
        m = zeros(3)
        m[1] = 1
        m[2] = 3/2
        m[3] = 7/3
        
        p₋ = [ R[j] >= 1.0 ? R[j]^(k*α₋ + ℓ₁ + 2) : R[j]^(k*α₊ + ℓ₁ + 2) for k ∈ 1:3 ]
        
        I_ = [
            2^(ℓ₂-1)*m[k]*p₋[k]^(-1)*(
                (logr₊ + π_(A[i+1]))^(ℓ₂) * I∞[i_coarse(i),ℓ₁+1,0+1,k] +
                I∞[i_coarse(i),ℓ₁+1,ℓ₂+1,k]
            ) for k ∈ 1:3 ]
        return (I_[1] + 3*I_[2] + 1*I_[3])/(2*π)
        
    elseif (ℓ₂ == 3 && ℓ₁ == 0)
        m = zeros(4)
        m[1] = 1
        m[2] = 3/2
        m[3] = 7/3
        m[4] = 15/4
        
        p₋ = [ R[j] >= 1.0 ? R[j]^(k*α₋ + ℓ₁ + 2) : R[j]^(k*α₊ + ℓ₁ + 2) for k ∈ 1:4 ]
        
        I_ = [
            2^(ℓ₂-1)*m[k]*p₋[k]^(-1)*(
                (logr₊ + π_(A[i+1]))^(ℓ₂) * I∞[i_coarse(i),ℓ₁+1,0+1,k] +
                I∞[i_coarse(i),ℓ₁+1,ℓ₂+1,k]
            ) for k ∈ 1:4 ]
        return (I_[1] + 7*I_[2] + 6*I_[3] + I_[4])/(2*π)
        
    elseif (ℓ₂ == 4 && ℓ₁ == 0)
        m = zeros(5)
        m[1] = 1
        m[2] = 3/2
        m[3] = 7/3
        m[4] = 15/4
        m[5] = 31/5
        
        p₋ = [ R[j] >= 1.0 ? R[j]^(k*α₋ + ℓ₁ + 2) : R[j]^(k*α₊ + ℓ₁ + 2) for k ∈ 1:5 ]
        
        I_ = [
            2^(ℓ₂-1)*m[k]*p₋[k]^(-1)*(
                (logr₊ + π_(A[i+1]))^(ℓ₂) * I∞[i_coarse(i),ℓ₁+1,0+1,k] +
                I∞[i_coarse(i),ℓ₁+1,ℓ₂+1,k]
            ) for k ∈ 1:5 ]
        return (I_[1] + 15*I_[2] + 25*I_[3] + 10*I_[4] + I_[5])/(2*π)
        
    else
        return 0.0
    end
end

function ∂(i,j,ℓ₁,ℓ₂)
    if (ℓ₁ == 0 && ℓ₂ == 0)
        return M[i,j]
    end
    
    S = [ U[i_coarse(i),ℓ₁+1,ℓ₂+1] ]
    
    if j == 1
        return S[1]
    end

    if ℓ₂ == 0
        append!(S,2*gamma((ℓ₁+1)/A[i]+1)/(2*π*R[j]))
    end
    if ℓ₁ == 0
        append!(S, osc[i_coarse(i),ℓ₂]/(2*π*A[i]^(ℓ₂)*R[j]) )
    end
    
    for n ∈ 1:3
        append!(S, series0(i,j,n-1,ℓ₁,ℓ₂) + remainder0(i,j,n-1,ℓ₁,ℓ₂) )
        append!(S, series∞(i,j,n,ℓ₁,ℓ₂) + remainder∞(i,j,n,ℓ₁,ℓ₂) )
    end
    
    return minimum(S)
end

############

D = zeros( length(A)-1, length(R)-1, 5, 5)

for ℓ₁ ∈ 1:4
    D[:,:,ℓ₁+1,1] = [ ∂(i,j,ℓ₁,0) for i ∈ 1:(length(A)-1), j ∈ 1:(length(R)-1) ]
end

for ℓ₂ ∈ 1:4
    D[:,:,1,ℓ₂+1] = [ ∂(i,j,0,ℓ₂) for i ∈ 1:(length(A)-1), j ∈ 1:(length(R)-1) ]
end

for ℓ₁ ∈ 1:2, ℓ₂ ∈ 1:2
    D[:,:,ℓ₁+1,ℓ₂+1] = [ ∂(i,j,ℓ₁,ℓ₂) for i ∈ 1:(length(A)-1), j ∈ 1:(length(R)-1) ]
end

h5write("derivatives_2d.h5","D",D)
h5write("derivatives_2d.h5","M",M)
