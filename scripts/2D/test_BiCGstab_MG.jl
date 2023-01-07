using LinearAlgebra, Plots, Printf, Statistics

# Only BiCGstab with diagonal preconditionner
# All fields are contained in 2D tables (no flatten/reshape business)

av4_arit(A) = 0.25.*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])   
av4_harm(A) = 1.0./( 0.25.*(1.0./A[1:end-1,1:end-1].+1.0./A[2:end,1:end-1].+1.0./A[1:end-1,2:end].+1.0./A[2:end,2:end]) )  

#--------------------------------------------------------------------#

function Residual2!( F, T, p, bcv )
    Δx=p.Δx; Δy=p.Δy; BCT=p.BCT; kv=p.kv; ncx=p.ncx; ncy=p.ncy; b=p.b
    for j=1:ncy
        for i=1:ncx
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

function DiagPC2!( Di, p )
    Δx=p.Δx; Δy=p.Δy; BCT=p.BCT; kv=p.kv; ncx=p.ncx; ncy=p.ncy
    for j=1:ncy
        for i=1:ncx
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

@views function Interp2D( Ηv, ilev1, ilev2, Ncx, Ncy, model, add )

    xmin=model.xmin; xmax=model.xmax; ymin=model.ymin; ymax=model.ymax

    Δx2 = (xmax-xmin)/(Ncx[ilev2]-1)
    Δx1 = (xmax-xmin)/(Ncx[ilev1]-1)
    Δy2 = (ymax-ymin)/(Ncy[ilev2]-1)
    Δy1 = (ymax-ymin)/(Ncy[ilev1]-1)

    for j2=1:Ncy[ilev2], i2=1:Ncx[ilev2]

        i1 = Int(floor((i2-1)*Δx2/Δx1)) + 1
        j1 = Int(floor((j2-1)*Δy2/Δy1)) + 1
        
        if i1==Ncx[ilev1] i1-=1 end
        if j1==Ncy[ilev1] j1-=1 end
        
        dW = (i2-1)*Δx2  - (i1-1)*Δx1
        wW = 1.0-dW/Δx1
        dS = (j2-1)*Δy2  - (j1-1)*Δy1
        wS = 1.0-dS/Δy1
   
        f  = wS *wW*Ηv[ilev1][i1  ,j1] + (1.0-wW)*     wS *Ηv[ilev1][i1+1,  j1] +
        (1.0-wS)*wW*Ηv[ilev1][i1,j1+1] + (1.0-wW)*(1.0-wS)*Ηv[ilev1][i1+1,j1+1]

        if add == 0
            Ηv[ilev2][i2,j2] = f 
        else
            Ηv[ilev2][i2,j2] += f
        end
    end
    return nothing
end

@views function Smooth(f, x, di, p, bc, niter)
    rel = 1.0
    for iter=1:niter
        Residual2!( f, x, p, bc )
        x .+= di.* f
        # x .= (1.0.-rel).*x .+ rel.*di.* f
    end
end

#--------------------------------------------------------------------#

function main(n)
    
    solver     = :BiCGstab
    xmin, xmax = -0.5, 0.5
    ymin, ymax = -0.5, 0.5
    ncx, ncy   = n*10, n*11
    rad        = 0.1
    Lx, Ly     = xmax-xmin, ymax-ymin
    Δx, Δy     = Lx/ncx, Ly/ncy
    kv         = ones(ncx+1, ncy+1)
    kc         = ones(ncx+0, ncy+0)
    xc  = LinRange(xmin+Δx/2, xmax-Δx/2, ncx)
    yc  = LinRange(ymin+Δy/2, ymax-Δy/2, ncy)
    xv  = LinRange(xmin, xmax, ncx+1)
    yv  = LinRange(ymin, ymax, ncy+1)
    for j=1:ncy+1, i=1:ncx+1
        if xv[i]^2+yv[j]^2<rad
            kv[i,j] = 1000.0
        end 
        # if (xv[i]+0.4)^2+(yv[j]-0.04)^2<rad/5
        #     kv[i,j] = .01
        # end
    end
    ism = 1
    for i=1:ism
        kc .= av4_harm(kv)
        kv[2:end-1,2:end-1] .= av4_harm(kc)
    end
    BCT = (W=0., E=1.0, S=1.0, N=2.0)
    T   = zeros(ncx,ncy)
    f   = zeros(ncx,ncy)
    b   = zeros(ncx,ncy)
    di  = zeros(ncx,ncy)

    
   
    Ncx  = Int.(floor.([ncx ncx/2 ncx/4 ]))
    Ncy  = Int.(floor.([ncy ncy/2 ncy/4 ]))
    Nlev = length(Ncx)
    Ηv   = Vector{Matrix{Float64}}(undef, Nlev)
    F    = Vector{Matrix{Float64}}(undef, Nlev)
    B    = Vector{Matrix{Float64}}(undef, Nlev)
    ΔT   = Vector{Matrix{Float64}}(undef, Nlev)
    Di   = Vector{Matrix{Float64}}(undef, Nlev)
    for i = 1:Nlev Ηv[i] = zeros(Ncx[i]+1, Ncy[i]+1)  end
    Ηv[1].= kv
    for i = 1:Nlev F[i]  = zeros(Ncx[i], Ncy[i])  end
    for i = 1:Nlev B[i]  = zeros(Ncx[i], Ncy[i])  end;
    for i = 1:Nlev ΔT[i] = zeros(Ncx[i], Ncy[i])  end
    for i = 1:Nlev Di[i] = zeros(Ncx[i], Ncy[i])  end
            Ηv[1].= kv
    model = ( xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
   

    for ilev=2:Nlev
        Interp2D( Ηv, ilev-1, ilev, Ncx.+1, Ncy.+1, model, 0 )
    end

    for ilev=Nlev-1:-1:1
        Interp2D( Ηv, ilev+1, ilev, Ncx.+1, Ncy.+1, model, 0 )
    end

    

    niter = 100
    Nlev  = 3
    ncyc  = 40

    params = (Δx=Lx/Ncx[1], Δy=Ly/Ncy[1], BCT=BCT, kv=Ηv[1], ncx=Ncx[1], ncy=Ncy[1], b=B[1] )
    Residual2!( F[1], ΔT[1], params, 1.0 )
    for Vcyc=1:ncyc

        for ilev=1:Nlev
            if ilev==1    
                params = (Δx=Lx/Ncx[ilev], Δy=Ly/Ncy[ilev], BCT=BCT, kv=Ηv[ilev], ncx=Ncx[ilev], ncy=Ncy[ilev], b=B[ilev] )
                DiagPC2!( Di[ilev], params )           
                @show mean(F[ilev])
                Smooth(F[ilev], ΔT[ilev], Di[ilev], params, 1.0, niter)
            else ilev>1
                ΔT[ilev] .= 0.0
                Interp2D( F, ilev-1, ilev, Ncx, Ncy, model, 0 )
                B[ilev] .= F[ilev]
                params = (Δx=Lx/Ncx[ilev], Δy=Ly/Ncy[ilev], BCT=BCT, kv=Ηv[ilev], ncx=Ncx[ilev], ncy=Ncy[ilev], b=B[ilev] )
                DiagPC2!( Di[ilev], params )
                Smooth(F[ilev], ΔT[ilev], Di[ilev], params, 0.0, niter)
            end
        
        end

        for ilev=Nlev:-1:2
            # if ilev<Nlev
                params = (Δx=Lx/Ncx[ilev], Δy=Ly/Ncy[ilev], BCT=BCT, kv=Ηv[ilev], ncx=Ncx[ilev], ncy=Ncy[ilev], b=F[ilev] )
                DiagPC2!( Di[ilev], params )                
                Smooth(F[ilev], ΔT[ilev], Di[ilev], params, 0.0, niter)
                Interp2D( ΔT, ilev, ilev-1, Ncx, Ncy, model, 1 )
                #  @show ilev
            # else
            #     params = (Δx=Lx/Ncx[ilev], Δy=Ly/Ncy[ilev], BCT=BCT, kv=Ηv[ilev], ncx=Ncx[ilev], ncy=Ncy[ilev], b=B[ilev] )
            #     Smooth(F[ilev], ΔT[ilev], Di[ilev], params, 1.0, niter)
            # end

        end

    end

    @show mean(F[1])

    p1=heatmap(log10.(Ηv[1])', aspect_ratio=1)
    p2=heatmap(log10.(Ηv[2])', aspect_ratio=1)
    p3=heatmap(log10.(Ηv[3])', aspect_ratio=1)
    p4=heatmap(xc, yc, ΔT[1]', aspect_ratio=1)
    display(plot(p1,p2,p3,p4))

    # noisy   = 0
    # tol     = 1e-10

    # Residual2!( F, T, params, 1.0 )
    # @show mean(F)

    # if solver==:BiCGstab
    #     p    = zeros(ncx,ncy)
    #     v    = zeros(ncx,ncy)
    #     h    = zeros(ncx,ncy)
    #     t    = zeros(ncx,ncy)
    #     s    = zeros(ncx,ncy)
    #     r̂0   = zeros(ncx,ncy)
    #     y    = zeros(ncx,ncy)
    #     z    = zeros(ncx,ncy)
    #     BiCG = (tol=tol, di=di, v=v, r̂0=r̂0, h=h, p=p, t=t, s=s, y=y, z=z, noisy=noisy, its=0)
    #     @time its = BiCGstab!( F, T, BiCG, params)
    # end
    # Residual2!( F, T, params, 1.0 )
    # @show (its, mean(F))

    # p1=heatmap(xc, yc, reshape(T,ncx,ncy)', aspect_ratio=1)
    # p2=heatmap(xv, yv, log10.(kv)', aspect_ratio=1)
    # display(plot(p1,p2))

end

main(2*5) #8*5