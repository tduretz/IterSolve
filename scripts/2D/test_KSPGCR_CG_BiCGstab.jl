using LinearAlgebra, Plots, Printf, Statistics

av4_arit(A) = 0.25.*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])   
av4_harm(A) = 1.0./( 0.25.*(1.0./A[1:end-1,1:end-1].+1.0./A[2:end,1:end-1].+1.0./A[1:end-1,2:end].+1.0./A[2:end,2:end]) )  

function Residual!( F, T, p, bcv )
    num=p.num; Δx=p.Δx; Δy=p.Δy; BCT=p.BCT; kv=p.kv; ncx=p.ncx; ncy=p.ncy; b=p.b
    for j=1:ncy
        for i=1:ncx
            eq    = num[i,j]
            if i==1   TW = bcv*2*BCT.W - T[eq]
            else      TW = T[num[i-1,j]]        end
            if i==ncx TE = bcv*2*BCT.E - T[eq] 
            else      TE = T[num[i+1,j]]        end
            if j==1   TS = bcv*2*BCT.S - T[eq]
            else      TS = T[num[i,j-1]]        end
            if j==ncy TN = bcv*2*BCT.N - T[eq] 
            else      TN = T[num[i,j+1]]        end
            TC    = T[eq]
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
            F[eq] = b[eq] - (qxE-qxW)/Δx - (qyN-qyS)/Δy
        end
    end
end

function ApplyPC!( F, T, p )
    num=p.num; Δx=p.Δx; Δy=p.Δy; BCT=p.BCT; kv=p.kv; ncx=p.ncx; ncy=p.ncy
    for j=1:ncy
        for i=1:ncx
            d  = 0.
            eq = num[i,j]
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
            F[eq] = T[eq]/d
        end
    end
end

function DiagPC!( Di, F, T, p )
    num=p.num; Δx=p.Δx; Δy=p.Δy; BCT=p.BCT; kv=p.kv; ncx=p.ncx; ncy=p.ncy
    for j=1:ncy
        for i=1:ncx
            d  = 0.
            eq = num[i,j]
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
            Di[eq] = 1.0/d
        end
    end
end

# function KSP_GCR_Jacobian!( Residual!, x::Vector{Float64}, M::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, eps::Float64, noisy::Int64, f::Vector{Float64}, v::Vector{Float64}, s::Vector{Float64}, val::Vector{Float64}, VV::Matrix{Float64}, SS::Matrix{Float64}, restart::Int64 )
@views function KSP_GCR_Jacobian!( r, x, KSP, params )
    tol=KSP.tol; restart=KSP.restart; val=KSP.val; s=KSP.s; v=KSP.v; SS=KSP.SS; VV=KSP.VV; noisy=KSP.noisy
    # Initialise
    val .= 0.0
    s   .= 0.0
    v   .= 0.0
    VV  .= 0.0
    SS  .= 0.0
    # KSP GCR solver                   
    norm_r, norm0   = 0.0, 0.0
    max_it          = 30*restart
    ncyc, its       = 0, 0
    i1, i2, success = 0, 0, 0
    # Initial residual
    Residual!( r, x, params, 1.0 )
    norm_r = sqrt(dot( r, r ) )#norm(v)norm(f)
    rnorm0 = norm_r
    # Solving procedure
    while its<max_it
        k = 1
        while (k<=restart)
            # Preconditionning
            # s .= r
            ApplyPC!( s, r, params )
            s .= .-s
            # Action of Jacobian on v
            Residual!( v, s, params, 0.0 )
            v .= .-v
            # Approximation of the Jv product
            for i = 1:k
                val[i] = dot(v, VV[:,i])
            end
            for i = 1:k
                v .= v .- val[i].*VV[:,i]
                s .= s .- val[i].*SS[:,i]
            end
            r_dot_v  = dot(r, v)
            nrm      = sqrt(dot(v, v))
            r_dot_v  = r_dot_v/nrm
            v       .= v./nrm
            s       .= s./nrm
            x       .= x .+ r_dot_v.*s
            r       .= r .- r_dot_v.*v
            norm_r   = norm(r)
            its     += 1;
            if (noisy==1) @printf("It. %04d: res. = %2.6e\n", its, norm_r/sqrt(length(r))) end
            # Check convergence
            if norm_r < tol*rnorm0 break; end
            VV[:,k] .= v;
            SS[:,k] .= s;
            k+=1
            # if k != restart break; end  # terminated restart loop earlier
        end
        ncyc += 1
        if norm_r/sqrt(length(r)) < tol break; end
    end
    if (noisy==1) @printf("[%1.4d] %1.6d KSP GCR Residual %1.12e %1.12e\n", ncyc, its, norm_r, norm_r/norm0); end
    return its
end
export KSP_GCR_Stokes!

@views function CG!( r, x, CG, params)
    tol=CG.tol; Ap=CG.Ap; p=CG.p; noisy=CG.noisy
    # Initial residual
    Residual!( r, x, params, 1.0 )
    p .= r
    rsold = dot(r, r)
    max_its = length(x)
    tot_its = max_its
    for its = 1:max_its
        Residual!( Ap, p, params, 0.0 ) #Ap = A * p;
        Ap  .*= -1.0
        α     = rsold / dot(p, Ap)
        x   .+= α .* p 
        r   .-= α .* Ap
        rsnew = dot(r, r)
        if (noisy==1) @printf("It. %04d: res. = %2.6e\n", its, sqrt(rsnew)/sqrt(length(r))) end
        if sqrt(rsnew)/sqrt(length(r)) < tol
            tot_its = its
            break
        end
        p .= r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end
    return tot_its
end

@views function BiCGstab!( r, x, BiCG, params)
    tol=BiCG.tol; di=BiCG.di; v=BiCG.v; r̂0=BiCG.r̂0; h=BiCG.h; p=BiCG.p; s=BiCG.s; t=BiCG.t; y=BiCG.y; z=BiCG.z; noisy=BiCG.noisy
    # Choose an arbitrary vector r̂0 such that (r̂0, r0) ≠ 0, e.g., r̂0 = r0
    Residual!( r, x, params, 1.0 )
    r̂0 .= r; v .= 0.0; p .= 0.0
    ρ, α, ω = 1.0, 1.0, 1.0
    max_its = length(x)
    tot_its = max_its
    for its = 1:max_its
        ω0, ρ0 = ω, ρ
        
        ρ  = dot(r̂0, r)
        ρ  = dot(r, r)
        β  = (ρ/ρ0)*(α/ω0)
        p .= r .+ β.* (p.−(ω0 .* v))
 
        # y .= p
        # ApplyPC!( y, p, params )
        y .= di.*p
        Residual!( v, y, params, 0.0 ) # v = A*p
        v .*= -1.0
        
        α  = ρ/dot(r̂0, v)
        h .= x .+ α.*y
        s .= r .- α.*v

        # z .= s
        # ApplyPC!( z, s, params )
        z .= di.*s
        Residual!( t, z, params, 0.0 ) # t = A*z
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
    
    # solver = :CG
    # solver = :KSPGCR
    solver = :BiCGstab
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
    neq = ncx*ncy
    num = reshape(1:neq,ncx,ncy)
    T   = zeros(neq)
    F   = zeros(neq)
    b   = zeros(ncx,ncy)
    di  = zeros(neq)
    params = (num=num, Δx=Δx, Δy=Δy, BCT=BCT, kv=kv, ncx=ncx, ncy=ncy, b=b )
    DiagPC!( di, F, T, params ) 

    restart = 100
    noisy   = 0
    tol     = 1e-10

    Residual!( F, T, params, 1.0 )
    @show mean(F)

    if solver==:KSPGCR
        val = zeros(neq)
        s   = zeros(neq)
        v   = zeros(neq)
        SS  = zeros(neq, restart)
        VV  = zeros(neq, restart)
        KSP = ( tol=tol, restart=restart, val=val, s=s, v=v, SS=SS, VV=VV, noisy=noisy )
        @time KSP_GCR_Jacobian!( F, T, KSP, params )
        @show mean(F)
    elseif solver==:CG
        Ap  = zeros(neq)
        p   = zeros(neq)
        CG  = (tol=tol, Ap=Ap, p=p, noisy=noisy)
        @time its = CG!( F, T, CG, params)
    elseif solver==:BiCGstab
        p    = zeros(neq)
        v    = zeros(neq)
        h    = zeros(neq)
        t    = zeros(neq)
        s    = zeros(neq)
        r̂0   = zeros(neq)
        y    = zeros(neq)
        z    = zeros(neq)
        BiCG = (tol=tol, di=di, v=v, r̂0=r̂0, h=h, p=p, t=t, s=s, y=y, z=z, noisy=noisy, its=0)
        @time its = BiCGstab!( F, T, BiCG, params)
    end
    Residual!( F, T, params, 1.0 )
    @show (its, mean(F))

    p1=heatmap(xc, yc, reshape(T,ncx,ncy)', aspect_ratio=1)
    p2=heatmap(xv, yv, log10.(kv)', aspect_ratio=1)
    display(plot(p1,p2))

end

main(2*5)