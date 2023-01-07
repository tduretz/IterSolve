using LinearAlgebra, Plots, Printf, Statistics

# Only BiCGstab with diagonal preconditionner
# All fields are contained in 2D tables (no flatten/reshape business)

av4_arit(A) = 0.25.*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])   
av4_harm(A) = 1.0./( 0.25.*(1.0./A[1:end-1,1:end-1].+1.0./A[2:end,1:end-1].+1.0./A[1:end-1,2:end].+1.0./A[2:end,2:end]) )  

#--------------------------------------------------------------------#

function Residual2!( F, T, p, bcv )
    Δx=p.Δx; Δy=p.Δy; BCT=p.BCT; kv=p.kv; ncx=p.ncx; ncy=p.ncy; b=p.b
    @inbounds for j=1:ncy
        @inbounds for i=1:ncx
            if i==1   TW = bcv*2*BCT.W - T[i,j]
            else      TW = T[i-1,j]        end
            if i==ncx TE = bcv*2*BCT.E - T[i,j] 
            else      TE = T[i+1,j]        end
            if j==1   TS = bcv*2*BCT.S - T[i,j]
            else      TS = T[i,j-1]        end
            if j==ncy TN = bcv*2*BCT.N - T[i,j] 
            else      TN = T[i,j+1]        end
            TC    = T[i,j]
            dTdxW = (TC-TW)/Δx
            dTdxE = (TE-TC)/Δx
            dTdyS = (TC-TS)/Δy
            dTduN = (TN-TC)/Δy
            kW    = 0.5*(kv[i,j] + kv[i,j+1])
            kE    = 0.5*(kv[i+1,j] + kv[i+1,j+1])
            kS    = 0.5*(kv[i,j] + kv[i+1,j])
            kN    = 0.5*(kv[i,j+1] + kv[i+1,j+1])
            # kW    = 1.0/(0.5*(1.0/kv[i,j]   + 1.0/kv[i,j+1]))
            # kE    = 1.0/(0.5*(1.0/kv[i+1,j] + 1.0/kv[i+1,j+1]))
            # kS    = 1.0/(0.5*(1.0/kv[i,j]   + 1.0/kv[i+1,j]))
            # kN    = 1.0/(0.5*(1.0/kv[i,j+1] + 1.0/kv[i+1,j+1]))
            qxW   = -kW*dTdxW
            qxE   = -kE*dTdxE
            qyS   = -kS*dTdyS
            qyN   = -kN*dTduN
            F[i,j] = b[i,j] - (qxE-qxW)/Δx - (qyN-qyS)/Δy
        end
    end
end

#--------------------------------------------------------------------#

function DiagPC2!( Di, F, T, p )
    Δx=p.Δx; Δy=p.Δy; BCT=p.BCT; kv=p.kv; ncx=p.ncx; ncy=p.ncy
    @inbounds for j=1:ncy
        @inbounds for i=1:ncx
            d  = 0.
            kW    = 0.5*(kv[i,j] + kv[i,j+1])
            kE    = 0.5*(kv[i+1,j] + kv[i+1,j+1])
            kS    = 0.5*(kv[i,j] + kv[i+1,j])
            kN    = 0.5*(kv[i,j+1] + kv[i+1,j+1])
            # kW    = 1.0/(0.5*(1.0/kv[i,j]   + 1.0/kv[i,j+1]))
            # kE    = 1.0/(0.5*(1.0/kv[i+1,j] + 1.0/kv[i+1,j+1]))
            # kS    = 1.0/(0.5*(1.0/kv[i,j]   + 1.0/kv[i+1,j]))
            # kN    = 1.0/(0.5*(1.0/kv[i,j+1] + 1.0/kv[i+1,j+1]))
            if i==1   d += kW/Δx^2 end
            if i==ncx d += kE/Δx^2 end 
            if j==1   d += kS/Δy^2 end
            if j==ncy d += kN/Δy^2 end
            d += (kW+kE)/Δx^2 + (kS+kN)/Δy^2
            Di[i,j] = 1.0/d
        end
    end
end

#--------------------------------------------------------------------#

@views function BiCGstab!( r, x, BiCG, params)
    tol=BiCG.tol; di=BiCG.di; v=BiCG.v; r̂0=BiCG.r̂0; h=BiCG.h; p=BiCG.p; s=BiCG.s; t=BiCG.t; y=BiCG.y; z=BiCG.z; noisy=BiCG.noisy
    # Choose an arbitrary vector r̂0 such that (r̂0, r0) ≠ 0, e.g., r̂0 = r0
    Residual2!( r, x, params, 1.0 )
    r̂0 .= r; v .= 0.0; p .= 0.0
    ρ, α, ω = 1.0, 1.0, 1.0
    max_its = length(x)
    tot_its = max_its
    for its = 1:max_its
        ω0, ρ0 = ω, ρ
        
        ρ  = dot(r̂0, r)
        β  = (ρ/ρ0)*(α/ω0)
        p .= r .+ β.* (p.−(ω0 .* v))
 
        # y .= p
        # ApplyPC!( y, p, params )
        y .= di.*p
        Residual2!( v, y, params, 0.0 ) # v = A*p
        v .*= -1.0
        
        α  = ρ/dot(r̂0, v)
        h .= x .+ α.*y
        s .= r .- α.*v

        # z .= s
        # ApplyPC!( z, s, params )
        z .= di.*s
        Residual2!( t, z, params, 0.0 ) # t = A*z
        t .*= -1.0

        ω  = dot2(t,s) # t's/t't
        x .= h .+ ω.*z
        r .= s .- ω.*t

        rsnew = dot(r, r)
        if (noisy==1) @printf("It. %04d: res. = %2.6e\n", its, sqrt(rsnew)/sqrt(length(r))) end
        if sqrt(rsnew)/sqrt(length(r)) < tol
            tot_its = its
            break
        end
    end
    return tot_its
end

#--------------------------------------------------------------------#

function dot(A, B)
    s = zero(promote_type(eltype(A), eltype(B)))
    for i in eachindex(A,B)
        s += A[i] * B[i]
    end
    return s
end

function dot2(A, B)
    s1 = zero(promote_type(eltype(A), eltype(B)))
    s2 = zero(promote_type(eltype(A), eltype(A)))
    for i in eachindex(A,B)
        s1 += A[i] * B[i]
        s2 += A[i] * A[i]
    end
    return s1/s2
end

#--------------------------------------------------------------------#

function main(n)
    
    solver     = :BiCGstab
    xmin, xmax = -0.5, 0.5
    ymin, ymax = -0.5, 0.5
    ncx, ncy   = n*10, n*11
    rad        = 0.1
    Δx, Δy     = (xmax-xmin)/ncx, (ymax-ymin)/ncy
    kv         = ones(ncx+1, ncy+1)
    kc         = ones(ncx+0, ncy+0)
    xc  = LinRange(xmin+Δx/2, xmax-Δx/2, ncx)
    yc  = LinRange(ymin+Δy/2, ymax-Δy/2, ncy)
    xv  = LinRange(xmin, xmax, ncx+1)
    yv  = LinRange(ymin, ymax, ncy+1)
    for j=1:ncy+1, i=1:ncx+1
        if xv[i]^2+yv[j]^2<rad
            kv[i,j] = 1000.00
        end 
        if (xv[i]+0.4)^2+(yv[j]-0.04)^2<rad/5
            kv[i,j] = .01
        end
    end
    ism = 1
    for i=1:ism
        kc .= av4_harm(kv)
        kv[2:end-1,2:end-1] .= av4_harm(kc)
    end
    BCT = (W=0., E=1.0, S=1.0, N=2.0)
    T   = zeros(ncx,ncy)
    F   = zeros(ncx,ncy)
    b   = zeros(ncx,ncy)
    di  = zeros(ncx,ncy)
    params = (Δx=Δx, Δy=Δy, BCT=BCT, kv=kv, ncx=ncx, ncy=ncy, b=b )
    DiagPC2!( di, F, T, params ) 

    restart = 100
    noisy   = 0
    tol     = 1e-10

    Residual2!( F, T, params, 1.0 )
    @show mean(F)

    if solver==:BiCGstab
        p    = zeros(ncx,ncy)
        v    = zeros(ncx,ncy)
        h    = zeros(ncx,ncy)
        t    = zeros(ncx,ncy)
        s    = zeros(ncx,ncy)
        r̂0   = zeros(ncx,ncy)
        y    = zeros(ncx,ncy)
        z    = zeros(ncx,ncy)
        BiCG = (tol=tol, di=di, v=v, r̂0=r̂0, h=h, p=p, t=t, s=s, y=y, z=z, noisy=noisy, its=0)
        @time its = BiCGstab!( F, T, BiCG, params)
    end
    Residual2!( F, T, params, 1.0 )
    @show (its, mean(F))

    p1=heatmap(xc, yc, reshape(T,ncx,ncy)', aspect_ratio=1)
    p2=heatmap(xv, yv, log10.(kv)', aspect_ratio=1)
    display(plot(p1,p2))

end

main(8*5)