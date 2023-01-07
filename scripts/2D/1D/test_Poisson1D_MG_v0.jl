using Plots
using Statistics: mean
using LinearAlgebra: norm

@views function Smooth1(f, x, di, p, bc, niter)
    rel = 1.0
    for iter=1:niter
        Residual1!( f, x, p, bc )
        x .= x .- rel.*di.* f
    end
end


function DiagPC1!( Di, p )
    Δx=p.Δx; BCT=p.BCT; kv=p.kv; ncx=p.ncx
    for i=1:ncx
        d  = 0.
        kW    = kv[i] 
        kE    = kv[i+1]
        if i==1   d += kW/Δx^2 end
        if i==ncx d += kE/Δx^2 end 
        d += (kW+kE)/Δx^2
        Di[i] = 1.0/d
    end
end

function Residual1!( F, T, p, bcv )
    Δx=p.Δx; BCT=p.BCT; kv=p.kv; ncx=p.ncx; b=p.b
    for i=1:ncx
        if i==1   TW = bcv*2*BCT.W - T[i]
        else      TW = T[i-1]        end
        if i==ncx TE = bcv*2*BCT.E - T[i] 
        else      TE = T[i+1]        end
        TC    = T[i]
        dTdxW = (TC-TW)/Δx
        dTdxE = (TE-TC)/Δx
        kW    = kv[i]
        kE    = kv[i+1]
        qxW   = -kW*dTdxW
        qxE   = -kE*dTdxE
        F[i]  = b[i] + (qxE-qxW)/Δx 
    end
end

@views function Interp1D( Ηv, ilev1, ilev2, Ncx, model, add )

    xmin=model.xmin; xmax=model.xmax

    Δx2 = (xmax-xmin)/(Ncx[ilev2]-1)
    Δx1 = (xmax-xmin)/(Ncx[ilev1]-1)

    for i2=1:Ncx[ilev2]

        i1 = Int(floor((i2-1)*Δx2/Δx1)) + 1
        
        if i1==Ncx[ilev1] i1-=1 end
        
        dW = (i2-1)*Δx2  - (i1-1)*Δx1
        wW = 1.0-dW/Δx1
   
        f  = wW*Ηv[ilev1][i1] + (1.0-wW)*     Ηv[ilev1][i1+1] 

        if add == 0
            Ηv[ilev2][i2] = f 
        else
            Ηv[ilev2][i2] += f
        end
    end
    return nothing
end

@views function Interp1D1( Ηv1, Ηv2, ilev1, ilev2, Ncx, model, add )

    xmin=model.xmin; xmax=model.xmax

    Δx2 = (xmax-xmin)/(Ncx[ilev2]-1)
    Δx1 = (xmax-xmin)/(Ncx[ilev1]-1)

    for i2=1:Ncx[ilev2]

        i1 = Int(floor((i2-1)*Δx2/Δx1)) + 1
        
        if i1==Ncx[ilev1] i1-=1 end
        
        dW = (i2-1)*Δx2  - (i1-1)*Δx1
        wW = 1.0-dW/Δx1
   
        f  = wW*Ηv1[i1] + (1.0-wW)*     Ηv1[i1+1] 

        if add == 0
            Ηv2[i2] = f 
        else
            Ηv2[i2] += f
        end
    end
    return nothing
end

function Poisson1D_MG(n)

    xmin, xmax = -0.5, 0.5
    Lx  = xmax-xmin
    ncx = n*1 + 1
    Δx  = (xmax-xmin)/ncx
    xc  = LinRange(xmin+Δx/2.0, xmax-Δx/2.0, ncx)
    xv  = LinRange(xmin, xmax, ncx+1)
    σ   = 0.05
    A   = 0.1
    kv  = ones(ncx+1) .+ A.*exp.(-xv.^2 ./ σ^2)
    b   = A.*exp.(-xc.^2 ./ σ^2)
    f   = zeros(ncx)
    u   = zeros(ncx)
    BCT = (W=0., E=0.0)
    tol = 1e-12

    # @show Ncx  = Int.(floor.([ncx]))
    @show Ncx  = Int.(floor.([ncx  ncx/2]))
    # @show Ncx  = Int.(floor.([ncx  ncx/2 ncx/4 ]))
    Nlev = length(Ncx)
    Ηv   = Vector{Vector{Float64}}(undef, Nlev)
    for i = 1:Nlev Ηv[i] = zeros(Ncx[i]+1)  end
    B     = Vector{Vector{Float64}}(undef, Nlev)
    for i = 1:Nlev B[i]  = zeros(Ncx[i])  end
    F     = Vector{Vector{Float64}}(undef, Nlev)
    for i = 1:Nlev F[i]  = zeros(Ncx[i])  end
    Di    = Vector{Vector{Float64}}(undef, Nlev)
    for i = 1:Nlev Di[i] = zeros(Ncx[i])  end
    Δu    = Vector{Vector{Float64}}(undef, Nlev)
    for i = 1:Nlev Δu[i] = zeros(Ncx[i])  end
    du    = Vector{Vector{Float64}}(undef, Nlev)
    for i = 1:Nlev du[i] = zeros(Ncx[i])  end
    Ηv[1] .= kv

    model = (xmin=xmin, xmax=xmax)

    for ilev=1:Nlev-1
        Interp1D( Ηv, ilev, ilev+1, Ncx.+1, model, 0 )
    end

    for ilev=Nlev:-1:2
        Interp1D( Ηv, ilev, ilev-1, Ncx.+1, model, 0 )
    end

    params = (Δx=Lx/Ncx[1], BCT=BCT, kv=Ηv[1], ncx=Ncx[1], b=b )
    DiagPC1!( Di[1], params )
    Residual1!( f, u, params, 1.0 )
    @show norm(f)/sqrt(length(f))
    Smooth1(f, u, Di[1], params, 1.0, 10000)
    @show norm(f)/sqrt(length(f))
    u .= 0.0
    B[1] .= f 
 
    #---------------------
    ncyc  = 20
    niter = 100

    for Vcyc=1:ncyc

        for ilev=1:Nlev

            Δu[ilev] .= 0.0 # reset corrections
            params = (Δx=Lx/Ncx[ilev], BCT=BCT, kv=Ηv[ilev], ncx=Ncx[ilev], b=B[ilev] )
            DiagPC1!( Di[ilev], params )
            Smooth1(F[ilev], Δu[ilev], Di[ilev], params, 0.0, niter)
                    
            if (ilev<Nlev)
                B[ilev+1] .= 0.0
               
                Interp1D1( F[ilev], B[ilev+1], ilev, ilev+1, Ncx, model, 0 )
                @show mean(B[ilev+1])
                @show mean(F[ilev])
            end
        end
    
        for ilev=Nlev:-1:1
    
            params = (Δx=Lx/Ncx[ilev], BCT=BCT, kv=Ηv[ilev], ncx=Ncx[ilev], b=B[ilev] )
            DiagPC1!( Di[ilev], params )
            Smooth1(F[ilev], Δu[ilev], Di[ilev], params, 0.0, niter)
                  
            if (ilev>1)
                du = zeros(Ncx[ilev-1])
                Interp1D1( Δu[ilev], du, ilev, ilev-1, Ncx, model, 0 )
                Δu[ilev-1] .+= du
                # Interp1D( Δu, ilev, ilev-1, Ncx, model, 1 )
            end
        end
        u .+= Δu[1]
        params = (Δx=Lx/Ncx[1], BCT=BCT, kv=Ηv[1], ncx=Ncx[1], b=b )
        Residual1!( f, u, params, 1.0 )
        B[1] .= f
        @show ( Vcyc, norm(f)/sqrt(length(f)) )
        if norm(f)/sqrt(length(f))<tol break end
    end

    #---------------------

    p = plot()
    for i = 1:1#Nlev
        # x = LinRange(xmin, xmax, Ncx[i]+1)
        # p = plot!(x, Ηv[i])
        x = LinRange(xmin+Δx/2, xmax-Δx/2, Ncx[i])
        # p = plot!(x, F[i])
        p = plot!(x, u)
    end
    display(p)

    @show Ncx

end

Poisson1D_MG(50)


