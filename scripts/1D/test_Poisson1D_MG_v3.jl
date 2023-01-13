# make it simpler
using Printf, Plots
import Statistics: mean
import LinearAlgebra: norm
import SparseArrays: sparse

#########################

function SOR2!(u, rhs, dx, k, iterMax, Sc, nx, ω, epsi, ndt, RP, R, ilev, noisy, inds )
    iter = 0; res = epsi*2; ncx = nx-1
    d = zeros(nx)
    # if ilev==1 
    #     ω = 4.0*(0.5 - 1.0/(nx/ilev))
    # else
        ω = 1.0
    # end
    i1=inds.i1; Δi=inds.Δi; i2=inds.i2
    nRB = 2
    while  (iter < iterMax) && (max(res) > epsi)
        for iRB=1:nRB # Red-black sweeps
            # Residuals
            for i=i1:Δi:i2
                if i>1 && i<nx # avoid Dirichlets
                    # if ilev==1
                        uS   = u[i-Δi]
                        uN   = u[i+Δi]
                    # end
                    # if ilev>1 && i==i1
                    #     uS   = -u[i]
                    # else
                    #     uS   = u[i-Δi]
                    # end
                    # if ilev>1 && i==i2
                    #     uN   = -u[i]
                    # else
                    #     uN   = u[i+Δi]
                    # end

                    qW   = k[i]*(u[i]-uS)/dx
                    qE   = k[i   ]*(uN-u[i])/dx
                    R[i] = (qE-qW)/dx - rhs[i]
                    d[i] = (k[i] .+ k[i])/dx/dx
                end
            end

            # nxC = length(i1:2^(ilev-1):i2)
            # if iseven(nxC) 
            #     seq = (i1+ilev*(iRB-1)):(ilev*(2)):(i2+ilev*(iRB-2))
            # else 
            #     seq = (i1+ilev*(iRB-1)):(ilev*(2)):(i2-ilev*(iRB-1))
            # end
            # isodd(nx)  ? true : seq = (i1+(iRB-1)):(ilev*(2)):(i2+(iRB-2))
            # iseven(nx) ? true : seq = (i1+ilev*(iRB-1)):(ilev*(2)):(i2-ilev*(iRB-1))
            isodd(nx)  ? true : seq = (i1+(iRB-1)):2:(i2+(iRB-2))
            iseven(nx) ? true : seq = (i1+ilev*(iRB-1)):2:(i2-ilev*(iRB-1))
            
            # if ilev==2 
                # @show (ilev, seq)
            # end
            
            for i = seq
                if i>1 && i<nx
                    u[i] += ω*R[i]/d[i]
                end
            end
        end
        # Resid check
        if mod(iter,ndt) == 1 && ilev == 1 && noisy
            q   = k.*diff(u,dims=1)/dx
            R  .= [0; diff(q,dims=1)/dx .- rhs[2:end-1]; 0]
            res = norm(R)/(length(R))
            @printf("Level %0d --- res = %2.2e\n", ilev, res)
        end
        iter += 1
    end
    # Residual
    q    = k.*diff(u,dims=1)/dx
    RP  .= rhs .- [0; diff(q,dims=1)/dx; 0]
    return iter
end

#########################

function GMG_classic!(u_MG, b, f, k, nlev, smoother, direct, NX, xmin, xmax, IT, Sc, damp, epsi, ndt, noisy)

    # Initialise
    DX      = (xmax-xmin)./(NX.-1)
    dx      = DX[1]
    XC = Vector{Vector{Float64}}(undef, nlev)
    XV = Vector{Vector{Float64}}(undef, nlev)
    K  = Vector{Vector{Float64}}(undef, nlev)
    B  = Vector{Vector{Float64}}(undef, nlev)
    U  = Vector{Vector{Float64}}(undef, nlev)
    R  = Vector{Vector{Float64}}(undef, nlev)
    Rd = Vector{Vector{Float64}}(undef, nlev)
    for ilev=1:nlev
        XV[ilev] = LinRange(xmin,xmax,NX[ilev])
        XC[ilev] = 0.5*(XV[ilev][1:end-1].+XV[ilev][2:end])
        U[ilev]  = zeros(NX[ilev]); R[ilev] = zeros(NX[ilev]); Rd[ilev] = zeros(NX[ilev]); B[ilev] = zeros(NX[ilev]); K[ilev]  = zeros(NX[ilev]-1);
    end
    B[1] .= f             # Initial RHS
    iter_tot = 0

    # From Fine to Coarse Restrict variable PDE coefficient 
    K[1] .= k
    for ilev=2:nlev
        Interp1D!(K[ilev-1], K[ilev], XC[ilev-1], XC[ilev], 0)
    end

    for itMG=1:100

        # V-cycle
        for ilev=1:nlev
            
            U[ilev] .= 0.0 # reset corrections
            
            if ilev == nlev && nlev>1 && direct
                U[ilev] .= DirectSolve( B[ilev], K[ilev], DX[ilev], NX[ilev] );
            else
                if smoother==:PT     iter =     PT!(U[ilev], B[ilev], DX[ilev], K[ilev], IT[ilev], Sc, NX[ilev], damp, epsi, ndt, R[ilev], Rd[ilev], ilev, noisy ) end
                if smoother==:Jacobi iter = Jacobi!(U[ilev], B[ilev], DX[ilev], K[ilev], IT[ilev], Sc, NX[ilev], damp, epsi, ndt, R[ilev], Rd[ilev], ilev, noisy ) end
                if smoother==:SOR    iter =    SOR!(U[ilev], B[ilev], DX[ilev], K[ilev], IT[ilev], Sc, NX[ilev], damp, epsi, ndt, R[ilev], Rd[ilev], ilev, noisy ) end
                if smoother==:SOR2        
                    inds=(i1=1, Δi=1, i2=NX[ilev])
                    iter = SOR2!(U[ilev], B[ilev], DX[ilev], K[ilev], IT[ilev], Sc, NX[ilev], damp, epsi, ndt, R[ilev], Rd[ilev], ilev, noisy, inds )
                end
            end

            if ilev==1 iter_tot = iter_tot + iter; end

            if (ilev<nlev)
                Interp1D!(R[ilev], B[ilev+1], XV[ilev], XV[ilev+1], 0)
            end
        end

        for ilev=nlev:-1:1

            if ilev == nlev && nlev>1 && direct
                U[ilev] .= DirectSolve( B[ilev], K[ilev], DX[ilev], NX[ilev] );
            else
                if smoother==:PT     iter =     PT!(U[ilev], B[ilev], DX[ilev], K[ilev], IT[ilev], Sc, NX[ilev], damp, epsi, ndt, R[ilev], Rd[ilev], ilev, noisy ) end
                if smoother==:Jacobi iter = Jacobi!(U[ilev], B[ilev], DX[ilev], K[ilev], IT[ilev], Sc, NX[ilev], damp, epsi, ndt, R[ilev], Rd[ilev], ilev, noisy ) end
                if smoother==:SOR    iter =    SOR!(U[ilev], B[ilev], DX[ilev], K[ilev], IT[ilev], Sc, NX[ilev], damp, epsi, ndt, R[ilev], Rd[ilev], ilev, noisy ) end
                if smoother==:SOR2        
                    inds=(i1=1, Δi=1, i2=NX[ilev])
                    iter = SOR2!(U[ilev], B[ilev], DX[ilev], K[ilev], IT[ilev], Sc, NX[ilev], damp, epsi, ndt, R[ilev], Rd[ilev], ilev, noisy, inds )
                end
            end
            if ilev==1 iter_tot = iter_tot + iter; end
            if (ilev>1)      
                Interp1D!( U[ilev], U[ilev-1], XV[ilev], XV[ilev-1],  1);
            end
        end
        u_MG .+= U[1]
        q    = k.*diff(u_MG,dims=1)/dx;
        f   .= b .- [0; diff(q,dims=1)/dx; 0];
        if (norm(f)/length(f) < 1*epsi) break; end
        @printf("it = %03d --- R = %2.2e\n", itMG, norm(f)/(length(f))); 
        B[1] .= f;
    end
    return iter_tot
end

#########################

function Interp1D!(F, C, xF, xC, add)
    nxF  = size(F  ,1);
    dxF  = xF[2]-xF[1]    
    dstx = xC.-xF[1] 
    ix   = Int.(floor.(dstx./dxF.-0.5)) .+ 1 # find index x
    ix[ix.<1]     .= 1
    ix[ix.>nxF-1] .= nxF-1         
    dxmF = ( xC .- xF[ix] )./dxF   # normalised distances x
    if add==1
        C .+= (1.0 .- dxmF).*F[ix   ] .+ (dxmF).*F[ix.+1]
    else
        C  .= (1.0 .- dxmF).*F[ix   ] .+ (dxmF).*F[ix.+1]
    end
end

#########################

function SOR!(u, rhs, dx, k, iterMax, Sc, nx, ω, epsi, ndt, RP, R, level, noisy )
    iter = 0; res = epsi*2; 
    while  (iter < iterMax) && (max(res) > epsi)
        for i=1:2 # Red-black sweeps
            # Residuals
            q   = k.*diff(u,dims=1)/dx
            R  .= [0; diff(q,dims=1)/dx .- rhs[2:end-1]; 0]
            # Update
            c   = (k[1:end-1] .+ k[2:end])/dx/dx;
            if i==1
                u[2:2:end-2]    .+= ω.*R[2:2:end-2]./c[1:2:end-1]
            else
                u[3:2:end-1]    .+= ω.*R[3:2:end-1]./c[2:2:end-0]
            end
        end
        # Resid check
        if mod(iter,ndt) == 1 && level == 1 && noisy
            q   = k.*diff(u,dims=1)/dx
            R  .= [0; diff(q,dims=1)/dx .- rhs[2:end-1]; 0]
            res = norm(R)/(length(R))
            @printf("Level %0d --- res = %2.2e\n", level, res)
        end
        iter += 1
    end
    # Residual
    q    = k.*diff(u,dims=1)/dx
    RP  .= rhs .- [0; diff(q,dims=1)/dx; 0]
    return iter
end

#########################

function Jacobi!(u, rhs, dx, k, iterMax, Sc, nx, damp, epsi, ndt, RP, R, level, noisy )
    iter = 0; res = epsi*2; 
    while  (iter < iterMax) && (max(res) > epsi)
        # Residuals
        q   = k.*diff(u,dims=1)/dx;
        R  .= [0; diff(q,dims=1)/dx .+ damp*R[2:end-1] .- rhs[2:end-1]; 0] ;
        # Update
        c   = (k[1:end-1] .+ k[2:end])/dx/dx;
        u[2:end-1] .= u[2:end-1] .+ R[2:end-1]./c;
        # Resid check
        if mod(iter,ndt) == 1 && level == 1 && noisy
            q   = k.*diff(u,dims=1)/dx;
            R  .= [0; diff(q,dims=1)/dx .+ damp*R[2:end-1] .- rhs[2:end-1]; 0] ;
            res = norm(R)/(length(R));
            @printf("Level %0d --- res = %2.2e\n", level, res)
        end
            iter = iter+1;
    end
    # Residual
    q    = k.*diff(u,dims=1)/dx;
    RP  .= rhs .- [0; diff(q,dims=1)./dx; 0];
    return iter
end

#########################

function PT!(u, rhs, dx, k, iterMax, Sc, nx, damp, epsi, ndt, RP, R, level, noisy )
    iter = 0; res = epsi*2
    RB = false
    while  (iter < iterMax) && (max(res) > epsi)
        # Iterative Timesteps
        dt    = dx.^2 ./(k)/4.1/Sc
        if RB
            for i=1:2 # Red-black sweeps
                # Residuals
                q   = k.*diff(u,dims=1)/dx
                R  .= [0; diff(q,dims=1)/dx .+ damp*R[2:end-1] .- rhs[2:end-1]; 0]
                # Update
                if i==1
                    u[2:2:end-2]    .+= 0.5.*(dt[1:2:end-2].+dt[2:2:end-1]).*R[2:2:end-2]
                else
                    u[3:2:end-1]    .+= 0.5.*(dt[2:2:end-1].+dt[3:2:end-0]).*R[3:2:end-1]
                end
            end
        else
            # Residuals
            q   = k.*diff(u,dims=1)/dx
            R  .= [0; diff(q,dims=1)/dx .+ damp*R[2:end-1] .- rhs[2:end-1]; 0] 
            # Update
            u[2:end-1] = u[2:end-1] .+ R[2:end-1].*( dt[1:end-1] .+ dt[2:end] ).*0.5
        end
        # Resid check
        if mod(iter,ndt) == 1 && level == 1  && noisy
            q   = k.*diff(u,dims=1)/dx
            R  .= [0; diff(q,dims=1)/dx .+ damp*R[2:end-1] .- rhs[2:end-1]; 0] 
            res = norm(R)/(length(R))
            @printf("Lev. %0d --- it. %06d --- res = %2.2e\n", level, iter, res)
        end
            iter = iter+1
    end
    # Residual
    q    = k.*diff(u,dims=1)/dx
    RP  .= rhs - [0; diff(q,dims=1)/dx; 0]
    return iter
end

#########################

function DirectSolve( b, k, dx, nx )
    # Direct solver
    NumUx = 1:nx;
    iUxC  = NumUx;
    iUxS  =  ones(size(iUxC)); iUxS[2:end-0] .= NumUx[1:end-1];
    iUxN  =  ones(size(iUxC)); iUxN[1:end-1] .= NumUx[2:end-0];
    cUxC  =  ones(size(iUxC)); cUxC[2:end-1] .= (k[2:end] .+ k[1:end-1])./dx^2;
    cUxS  = zeros(size(iUxC)); cUxS[2:end-1] .=-(            k[1:end-1])./dx^2;
    cUxN  = zeros(size(iUxC)); cUxN[2:end-1] .=-(k[2:end]              )./dx^2;
    I     = [iUxC[:];iUxC[:];iUxC[:]];
    J     = [iUxS[:];iUxC[:];iUxN[:]];
    V     = [cUxS[:];cUxC[:];cUxN[:]];
    M     = sparse(I,J,V);
    u_dir =-M\b;
    return u_dir
end

#########################

function Poisson1D_MG_v3(n)

    # Domain
    nx   = n*101
    ncx  = nx-1
    xmin = -0.5
    xmax =  0.5
    dx   =  (xmax-xmin)/(ncx)
    # Forcing term (RHS)
    Amp  = 0.1
    Sig  = 0.05
    xc  = LinRange(xmin+dx/2.0, xmax-dx/2.0, ncx)
    xv  = LinRange(xmin, xmax, ncx+1)
    b    = zeros(nx,1) .+ Amp*exp.(-(xv).^2 /Sig^2); 
    # Variable coefficient
    k    = ones(ncx,1) .+ 0.0*Amp*exp.(-(xc).^2 /Sig^2); 
    # Initial solution
    u    = zeros(nx,1);

    # Residual
    q    = k.*diff(u,dims=1)/dx;
    f    = b .- [0; diff(q,dims=1)./dx; 0];
    f0   = copy(f)
    @printf( "Initial residual: %2.2e\n", norm(f)/((length(f))));
    @printf( "mean residual: %2.2e\n", mean(f));

    #-------------------------------------------------------------------------%

    # Direct Solve
    @printf("\nDirect Solve (UMFPACK)\n");
    u_dir = DirectSolve( b, k, dx, nx );
    q     = k.*diff(u_dir,dims=1)/dx;
    RP    = b .- [0; diff(q,dims=1)./dx; 0];
    @printf("Converged down to R = %2.2e\n", norm(RP)/(length(RP))); 

    #-------------------------------------------------------------------------%

    # # Jacobi solve
    # @printf("\nJacobi solver\n"); 
    # u_Jac   = zeros(size(u))
    # R       = zeros(size(u))
    # Rd      = zeros(size(u))
    # noisy   = true
    # dmp     = 4.0
    # Sc      = 0.5
    # damp    = 0*(1.0-dmp/nx)
    # iterMax = 2e4
    # epsi    = 1e-11
    # ndt     = 1000
    # level   = 1
    # u      .= 0.0
    # iter    = Jacobi!(u_Jac, b, dx, k, iterMax, Sc, nx, 1*damp, epsi, ndt, R, Rd, level, noisy );
    # @printf("Converged in %02d iterations down to R = %2.2e\n", iter, norm(RP)/(length(RP))); 

    #-------------------------------------------------------------------------%

    # PT solve
    @printf("\nPT solver with damping\n"); 
    u_PT    = zeros(size(u))
    R       = zeros(size(u))
    Rd      = zeros(size(u))
    noisy   = true
    dmp     = 4.0
    Sc      = 0.5
    damp    = 1*(1-dmp/nx)
    iterMax = 6e4
    epsi    = 1e-11
    ndt     = 1000
    level   = 1
    u      .= 0.0
    iter = PT!(u_PT, b, dx, k, iterMax, Sc, nx, damp, epsi, ndt, R, Rd, level, noisy )
    @printf("Converged in %02d iterations down to R = %2.2e\n", iter, norm(RP)/(length(RP)))


    #-------------------------------------------------------------------------%

    # SOR solve
    @printf("\nRed-Black SOR solver\n"); 
    u_SOR   = zeros(size(u))
    R       = zeros(size(u))
    Rd      = zeros(size(u))
    noisy   = true
    dmp     = 4.0
    Sc      = 0.5
    damp    = 4.0*(0.5 - 1.0/nx)
    iterMax = 2e4
    epsi    = 1e-11
    ndt     = 1000
    level   = 1
    u      .= 0.0
    # iter    = SOR!(u_SOR, b, dx, k, iterMax, Sc, nx, damp, epsi, ndt, R, Rd, level, noisy );
    inds    = (i1=1, Δi=1, i2=nx)
    iter    = SOR2!(u_SOR, b, dx, k, iterMax, Sc, nx, damp, epsi, ndt, R, Rd, level, noisy, inds );
    @printf("Converged in %02d iterations down to R = %2.2e\n", iter, norm(RP)/(length(RP))); 

    #-------------------------------------------------------------------------%

    # Multigrid solver
    @printf("\nGMG solver (classic)\n")
    noisy   = false
    u_MG    = zeros(size(u)) # Initial solution
    dmp     = 4.0
    Sc      = 0.5
    damp    = 1.0
    epsi    = 1e-13
    ndt     = 1000
    @show NX      = reverse(Int.(floor.(LinRange(40, nx, n+1))))
    IT      = 100 .* ones(size(NX))
    nlev    = length(NX)
    direct  = true
    # smooth = :PT
    # smooth = :Jacobi
    smooth  = :SOR
    iter    = GMG_classic!(u_MG, b, f, k, nlev, smooth, direct, NX, xmin, xmax, IT, Sc, damp, epsi, ndt, noisy)
    @printf("Converged in %02d iterations down to R = %2.2e\n", iter, norm(f)/(length(f))); 

    #-------------------------------------------------------------------------%

    # # Multigrid solver
    # @printf("\nGMG solver (modified)\n")
    # u_MG = zeros(size(u)) # Initial solution
    # noisy   = false
    # dmp     = 4.0
    # Sc      = 0.5
    # damp    = 1.0
    # epsi    = 1e-13
    # ndt     = 1000
    # @show NX = [nx nx]
    # IT      = 100 .* ones(size(NX))
    # nlev    = length(NX)
    # # smooth = :PT
    # # smooth = :Jacobi
    # smooth  = :SOR2

    # # Initialise
    # DX      = (xmax-xmin)./(NX.-1)
    # XC = Vector{Vector{Float64}}(undef, nlev)
    # XV = Vector{Vector{Float64}}(undef, nlev)
    # K  = Vector{Vector{Float64}}(undef, nlev)
    # B  = Vector{Vector{Float64}}(undef, nlev)
    # U  = Vector{Vector{Float64}}(undef, nlev)
    # R  = Vector{Vector{Float64}}(undef, nlev)
    # Rd = Vector{Vector{Float64}}(undef, nlev)
    # for ilev=1:nlev
    #     XV[ilev] = LinRange(xmin,xmax,NX[ilev])
    #     XC[ilev] = 0.5*(XV[ilev][1:end-1].+XV[ilev][2:end])
    #     U[ilev]  = zeros(NX[ilev]); R[ilev] = zeros(NX[ilev]); Rd[ilev] = zeros(NX[ilev]); B[ilev] = zeros(NX[ilev]); K[ilev]  = zeros(NX[ilev]-1);
    # end
    # B[1] .= f0             # Initial RHS
    # u_MG = zeros(size(u)) # Initial solution
    # iter_tot = 0

    # for itMG=1:100

    #     # V-cycle
    #     for ilev=1:nlev
            
    #         U[ilev] .= 0.0 # reset corrections
            
    #         if ilev==1
    #             # Smooth
    #             inds = (i1=1, Δi=1, i2=NX[ilev])
    #             iter = SOR2!(U[ilev], B[ilev], DX[ilev], k, IT[ilev], Sc, NX[ilev], damp, epsi, ndt, R[ilev], Rd[ilev], ilev, noisy, inds )
               
    #         end
    #         if ilev==2
    #             # Smooth
    #             inds = (i1=1, Δi=2, i2=NX[ilev])
    #             iter = SOR2!(U[ilev], B[ilev], DX[ilev]*2, k, IT[ilev], Sc, NX[ilev], damp, epsi, ndt, R[ilev], Rd[ilev], ilev, noisy, inds )
               
    #             inds = (i1=2, Δi=2, i2=NX[ilev]-1)
    #             iter = SOR2!(U[ilev], B[ilev], DX[ilev]*2, k, IT[ilev], Sc, NX[ilev], damp, epsi, ndt, R[ilev], Rd[ilev], ilev, noisy, inds )
    #         end

    #         if ilev==1 iter_tot = iter_tot + iter; end
    #         if (ilev<nlev)
    #             B[ilev+1] .= R[ilev]
    #         end
            
    #     end

    #     for ilev=nlev:-1:1

    #         if ilev==1
    #             # Smooth
    #             inds = (i1=1, Δi=1, i2=NX[ilev])
    #             iter = SOR2!(U[ilev], B[ilev], DX[ilev], k, IT[ilev], Sc, NX[ilev], damp, epsi, ndt, R[ilev], Rd[ilev], ilev, noisy, inds )
    #         end
    #         if ilev==2
    #             inds = (i1=1, Δi=2, i2=NX[ilev])
    #             iter = SOR2!(U[ilev], B[ilev], DX[ilev]*2, k, IT[ilev], Sc, NX[ilev], damp, epsi, ndt, R[ilev], Rd[ilev], ilev, noisy, inds )
               
    #             inds = (i1=2, Δi=2, i2=NX[ilev]-1)
    #             iter = SOR2!(U[ilev], B[ilev], DX[ilev]*2, k, IT[ilev], Sc, NX[ilev], damp, epsi, ndt, R[ilev], Rd[ilev], ilev, noisy, inds )
    #         end

    #         if ilev==1 iter_tot = iter_tot + iter; end
    #         if (ilev>1)
    #             # @show U[ilev] 
    #             U[ilev-1] .+= U[ilev]      
    #         end
    #     end
    #     u_MG .+= U[1]
    #     q    = k.*diff(u_MG,dims=1)/dx;
    #     f    = b .- [0; diff(q,dims=1)/dx; 0];
    #     if (norm(f)/length(f) < 1*epsi) break; end
    #     @printf("it = %03d --- R = %2.2e\n", itMG, norm(f)/(length(f))); 
    #     B[1] .= f;
    # end
    # @printf("Converged in %02d iterations down to R = %2.2e\n", iter_tot, norm(f)/(length(f))); 

    #-------------------------------------------------------------------------%

    p = plot()
    p = plot!(xv, u_dir, label="direct")
    p = plot!(xv, u_PT,  label="PT (damped)")
    # p = plot!(xv, u_Jac, label="Jacobi")
    p = plot!(xv, u_SOR, label="GS Red-black")
    p = plot!(xv, u_MG,  label="GMG (SOR)")
    display(p)

end
    
Poisson1D_MG_v3(2)