using QuadGK, HDF5, SpecialFunctions

Δ = .01
A = .5:Δ:1.95
A = [ α for α ∈ A ]

filename = "integrals_2d.h5"
h5write(filename,"A",A)

integral_tolerance = 1e-16

nmax = 3                # Maximum number of terms for the series expansions
π_(α) = π/(2*max(α,1))

## Polynomials pℓ given by ∂ℓ/∂α^ℓ [ exp(-t^α) ] = log(t)^ℓ * pℓ(t^α) * exp(-t^α)
p(t) = [1.0,
        -t,
        t^2 - t,
        -t^3 + 3*t^2 - t,
        t^4 - 6*t^3 + 7*t^2 - t,
        -t^5 + 10*t^4 - 25*t^3 + 15*t^2 - t]


## Hankel function of order 0 and its derivatives
H(z) = [
        hankelh1(0,z),
        -hankelh1(1,z),
        hankelh1(1,z)/z - hankelh1(0,z),
        (1 - 2/z^2)*hankelh1(1,z) + hankelh1(0,z)/z,
        (1 - 3/z^2)*hankelh1(0,z) - 2*(z^2-3)*hankelh1(1,z)/z^3
       ]

## Error term coefficients for r -> 0

R0 = zeros(length(A)-1, 5,5,nmax+1)

for ℓ₁ ∈ 0:4, ℓ₂ ∈ 0:4, n ∈ 0:nmax
    z = 2*n+4+2*ceil(ℓ₁/2)
    R0[:,ℓ₁+1,ℓ₂+1,n+1] = [ quadgk(t -> abs(log(t))^(ℓ₂) *
                     max(t^(z/A[i]),t^(z/A[i+1])) * t^(-1) *
                     abs(p(t)[ℓ₂+1]) * exp(-t),
                     0, Inf, rtol=integral_tolerance)[1] for i ∈ 1:(length(A)-1) ]
end

h5write(filename,"R0",R0)

## Error term coefficients for r -> ∞

I∞ = zeros(length(A)-1, 5, 5, nmax+3)

for ℓ₁ ∈ 0:4, ℓ₂ ∈ 0:4, k ∈ 1:(nmax+3)
    I∞[:,ℓ₁+1,ℓ₂+1,k] = [ quadgk(τ -> abs(H(exp(im*π_(A[i+1]))*τ)[ℓ₁+1]) *
             max(τ^(k*A[i]),τ^(k*A[i+1])) * τ^(ℓ₁+1) *
             abs(log(τ))^(ℓ₂),
             0, Inf, rtol = integral_tolerance)[1]
       for i ∈ 1:(length(A)-1) ]
end

h5write(filename,"I∞",I∞)

## Uniform bounds

U = [
     (2*π)^(-1) * A[i]^(-ℓ₂-1) *
     quadgk(t -> abs(log(t))^(ℓ₂) * abs(p(t)[ℓ₂+1]) *
            max(t^((ℓ₁+2)/A[i]),t^((ℓ₁+2)/A[i+1])) * t^(-1) *
            exp(-t),
            0, Inf, rtol=integral_tolerance)[1]
     for i ∈ 1:(length(A)-1), ℓ₁ ∈ 0:4, ℓ₂ ∈ 0:4
     ]

h5write(filename,"U",U)

## Oscillatory bound coefficients. Unlike in the 1D case, these depend on α

osc = zeros(length(A)-1,4)

osc[:,1] = [ quadgk(t -> max(t^(1/A[i]),t^(1/A[i+1]))*abs(log(t)*t - log(t) - 1)*exp(-t) , 0, Inf, rtol=integral_tolerance)[1] for i ∈ 1:(length(A)-1)]
osc[:,2] = [ quadgk(t -> max(t^(1/A[i]),t^(1/A[i+1]))*abs(log(t))*abs(log(t)*t^2 - 3*log(t)*t - 2*t + log(t) + 2)*exp(-t) , 0, Inf, rtol=integral_tolerance)[1] for i ∈ 1:(length(A)-1) ]
osc[:,3] = [ quadgk(t -> max(t^(1/A[i]),t^(1/A[i+1]))*abs(log(t))^2*abs(log(t)*t^3 - 6*log(t)*t^2 - 3*t^2 + 7*log(t)*t + 9*t - log(t) - 3)*exp(-t) , 0, Inf, rtol=integral_tolerance)[1] for i ∈ 1:(length(A)-1) ]
osc[:,4] = [ quadgk(t -> max(t^(1/A[i]),t^(1/A[i+1]))*abs(log(t))^3*abs(log(t)*t^4 - 10*log(t)*t^3 - 4*t^3 + 25*log(t)*t^2 + 24*t^2 - 15*log(t)*t - 28*t + log(t) + 4)*exp(-t) , 0, Inf, rtol=integral_tolerance)[1] for i ∈ 1:(length(A)-1) ]

h5write(filename,"osc",osc)
