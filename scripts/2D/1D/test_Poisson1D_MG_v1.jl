# using matlab working code
using Plots
using Statistics: mean
using LinearAlgebra: norm

function Prolongate(Fine, Coarse, xF, xC, itp)
    nxF = size(Fine  ,1); 
    nxC = size(Coarse,1); 
    dxC = xC[2]-xC[1];    
    C   = Coarse[:];      F   = Fine[:];    sumW = zeros(size(F))
    dstx = xF.-xC[1]; 
    ix   = Int.(floor.(dstx./dxC.-0.5)) .+ 1; 
    ix[ix.<1] .= 1; ix[ix.>nxC.-1] .= nxC.-1; # find index x
    dxmC = (xF[:] .- xC[ix]  )./dxC;                                              # normalised distances x
    F .= F .+ (1.0 .- dxmC).*C[ix   ]; sumW .= sumW .+ (1.0 .- dxmC); # contrib. SW
    F .= F .+ (       dxmC).*C[ix.+1]; sumW .= sumW .+ (       dxmC); # contrib. SE
    F .= F ./ sumW; 
    Fine .= F
end

function Restrict(Fine, Coarse, xF, xC, itp)
    nxF = size(Fine  ,1); 
    nxC = size(Coarse,1); 
    dxF = xF[2]-xF[1];    
    C   = Coarse[:];      F   = Fine[:];    sumW = zeros(size(C))
    dstx = xC.-xF[1]; 
    ix   = Int.(floor.(dstx./dxF.-0.5)) .+ 1; 
    ix[ix.<1] .= 1; ix[ix.>nxF-1] .= nxF-1; # find index x
    dxmF = ( xC .- xF[ix] )./dxF;                                                   # normalised distances x
    C .= C .+ (1.0 .- dxmF).*F[ix   ]; sumW .= sumW .+ (1.0 .- dxmF); # contrib. SW
    C .= C .+ (       dxmF).*F[ix.+1]; sumW .= sumW .+ (       dxmF); # contrib. SE
    C .= C ./ sumW; 
    Coarse.=C
end

function GaussSeidel_solver(u, rhs, dx, k, iterMax, Sc, nx, damp, epsi, ndt, R, level )

    iter = 0; res = epsi*2;
    while  (iter < iterMax) && (max(res) > epsi)
        for i=1:2
            # Residuals
            q   = k.*diff(u,dims=1)/dx;
            R   = [0; diff(q,dims=1)/dx .+ damp*R[2:end-1] .- rhs[2:end-1]; 0] ;
            # Update
            c   = 2/dx/dx;
            if i==1
                u[2:2:end-2] .= u[2:2:end-2] .+ R[2:2:end-2]./c;
            else
                u[3:2:end-1] .= u[3:2:end-1] .+ R[3:2:end-1]./c;
            end
        end
        # Resid check
        if mod(iter,ndt) == 1 && level == 1
            q   = k.*diff(u,dims=1)/dx;
            R   = [0; diff(q,dims=1)/dx .+ damp*R[2:end-1] .- rhs[2:end-1]; 0] ;
            res = norm(R)/(length(R));
            @printf("Level %0d --- res = %2.2e\n", level, res)
        end
        iter = iter+1;
    end
    # Residual
    q    = k.*diff(u,dims=1)/dx;
    RP   = rhs .- [0; diff(q,dims=1)/dx; 0];
    return (u, R, RP, iter)
end

function Jacobi_solver(u, rhs, dx, k, iterMax, Sc, nx, damp, epsi, ndt, R, level )

    iter = 0; res = epsi*2; 
    while  (iter < iterMax) && (max(res) > epsi)
        # Residuals
        q   = k.*diff(u,dims=1)/dx;
        R   = [0; diff(q,dims=1)/dx .+ damp*R[2:end-1] .- rhs[2:end-1]; 0] ;
        # Update
        c   = 2/dx/dx;
        u[2:end-1] = u[2:end-1] + R[2:end-1]./c;
        # Resid check
        if mod(iter,ndt) == 1 && level == 1
            q   = k.*diff(u,dims=1)/dx;
            R   = [0; diff(q,dims=1)/dx .+ damp*R[2:end-1] .- rhs[2:end-1]; 0] ;
            res = norm(R)/(length(R));
            @printf("Level %0d --- res = %2.2e\n", level, res)
        end
            iter = iter+1;
    end
    # Residual
    q    = k.*diff(u,dims=1)/dx;
    RP   = rhs .- [0; diff(q,dims=1)./dx; 0];
    return (u, R, RP, iter)
end

function PT_solver(u, rhs, dx, k, iterMax, Sc, nx, damp, epsi, ndt, R, level )

    iter = 0; res = epsi*2;
    while  (iter < iterMax) && (max(res) > epsi)
        # Iterative Timesteps
        dt    = dx.^2 ./(k)/4.1/Sc;
        # Residuals
        q   = k.*diff(u,dims=1)/dx;
        R   = [0; diff(q,dims=1)/dx .+ damp*R[2:end-1] .- rhs[2:end-1]; 0] ;
        # Update
        u[2:end-1] = u[2:end-1] .+ R[2:end-1,:].*( dt[1:end-1,:] .+ dt[2:end,:] ).*0.5;
        # Resid check
        if mod(iter,ndt) == 1 && level == 1
            q   = k.*diff(u,dims=1)/dx;
            R   = [0; diff(q,dims=1)/dx .+ damp*R[2:end-1] .- rhs[2:end-1]; 0] ;
            res = norm(R)/(length(R));
            @printf("Lev. %0d --- it. %06d --- res = %2.2e\n", level, iter, res)
        end
            iter = iter+1;
    end
    # Residual
    q    = k.*diff(u,dims=1)/dx;
    RP   = rhs - [0; diff(q,dims=1)/dx; 0];
    return (u, R, RP, iter)
end

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

function Poisson1D_MG_v2(n)

    # Domain
    nx   = 100;
    ncx  = nx-1;
    xmin = -0.5;
    xmax =  0.5;
    dx   =  (xmax-xmin)/(ncx);
    # Forcing term (RHS)
    Amp  = 0.1;
    Sig  = 0.05;
    xc  = LinRange(xmin+dx/2.0, xmax-dx/2.0, ncx)
    xv  = LinRange(xmin, xmax, ncx+1)
    b    = zeros(nx,1) .+ Amp*exp.(-(xv).^2 /Sig^2); 
    # Variable coefficient
    k    = ones(ncx,1) .+ Amp*exp.(-(xc).^2 /Sig^2); 
    # Initial solution
    u    = zeros(nx,1);

    # Residual
    q    = k.*diff(u,dims=1)/dx;
    f    = b .- [0; diff(q,dims=1)./dx; 0];
    @printf( "Initial residual: %2.2e\n", norm(f)/(sqrt(length(f))));
    @printf( "mean residual: %2.2e\n", mean(f));

    # %-------------------------------------------------------------------------%

    # Direct Solve
    @printf("\nDirect Solve (UMFPACK)\n");
    u_dir = DirectSolve( b, k, dx, nx );
    q     = k.*diff(u_dir,dims=1)/dx;
    RP    = b .- [0; diff(q,dims=1)./dx; 0];
    @printf("Converged down to R = %2.2e\n", norm(RP)/(length(RP))); 

    # %-------------------------------------------------------------------------%

    # PT solve
    @printf("\nPT solver with damping\n"); 
    R       = zeros(size(u));
    dmp     = 4.0;
    Sc      = 0.5;
    damp    = 1*(1-dmp/nx);
    iterMax = 6e4;
    epsi    = 1e-13;
    ndt     = 1000;
    level   = 1;
    u      .= 0.0
    (u_PT, R, RP, iter) = PT_solver(u, b, dx, k, iterMax, Sc, nx, damp, epsi, ndt, R, level );
    @printf("Converged in %02d iterations down to R = %2.2e\n", iter, norm(RP)/(length(RP))); 

    # %-------------------------------------------------------------------------%

    # Jacobi solve
    @printf("\nJacobi solver with damping\n"); 
    R       = zeros(size(u));
    dmp     = 4.0;
    Sc      = 0.5;
    damp    = 1-dmp/nx;
    iterMax = 2e4;
    epsi    = 1e-13;
    ndt     = 1000;
    level   = 1;
    u      .= 0.0
    (u_jac, R, RP, iter) = Jacobi_solver(u, b, dx, k, iterMax, Sc, nx, 1*damp, epsi, ndt, R, level );
    @printf("Converged in %02d iterations down to R = %2.2e\n", iter, norm(RP)/(length(RP))); 

    # %-------------------------------------------------------------------------%

    # Gauss-Seidel solve
    @printf("\nRed-Black Gauss-Seidel solver with damping\n"); 
    R       = zeros(size(u));
    dmp     = 4.0;
    Sc      = 0.5;
    damp    = 1-dmp/nx;
    iterMax = 2e4;
    epsi    = 1e-13;
    ndt     = 1000;
    level   = 1;
    u      .= 0.0
    (u_GS, R, RP, iter) = GaussSeidel_solver(u, b, dx, k, iterMax, Sc, nx, damp, epsi, ndt, R, level );
    @printf("Converged in %02d iterations down to R = %2.2e\n", iter, norm(RP)/(length(RP))); 

    # %-------------------------------------------------------------------------%

    # Multigrid solver
    @printf("\nGMG solver\n");
    dmp     = 4.0;
    Sc      = 0.5;
    damp    = 0*(1-dmp/nx);
    epsi    = 1e-13;
    ndt     = 1000;
    NX      = [100  40];
    IT      = [100 100];
    nlev    = length(NX);
    vcoef   = 1.0;

    # Initialise
    DX      = (xmax-xmin)./(NX.-1);
    XC = Vector{Vector{Float64}}(undef, nlev)
    XV = Vector{Vector{Float64}}(undef, nlev)
    K  = Vector{Vector{Float64}}(undef, nlev)
    B  = Vector{Vector{Float64}}(undef, nlev)
    U  = Vector{Vector{Float64}}(undef, nlev)
    R  = Vector{Vector{Float64}}(undef, nlev)
    Rd = Vector{Vector{Float64}}(undef, nlev)
    for ilev=1:nlev
        XV[ilev] = LinRange(xmin,xmax,NX[ilev]);
        XC[ilev] = 0.5*(XV[ilev][1:end-1].+XV[ilev][2:end]);
        U[ilev]  = zeros(NX[ilev]); R[ilev] = zeros(NX[ilev]); Rd[ilev] = zeros(NX[ilev]); B[ilev] = zeros(NX[ilev]); K[ilev]  = zeros(NX[ilev]-1);
    end
    B[1] .= f;     # Initial RHS
    u_mg = zeros(size(u)); # Initial solution
    iter_tot = 0;

    # From Fine to Coarse Restrict variable PDE coefficient 
    K[1] .= k;
    for ilev=2:nlev
        K[ilev] .= 0.
        K[ilev] .= Restrict(K[ilev-1], K[ilev], XC[ilev-1], XC[ilev], 1)
    end

    for itMG=1:100

        # V-cycle
        for ilev=1:nlev
            #  damp    = 1-dmp/NX[ilev] # Turn off damping for MG
   
            
            U[ilev] .= 0.0; # reset corrections
            
             if ilev == nlev && nlev>1
                 U[ilev] = DirectSolve( B[ilev], K[ilev], DX[ilev], NX[ilev] );
             else
                (U[ilev], Rd[ilev], R[ilev], iter) = PT_solver(U[ilev], B[ilev], DX[ilev], K[ilev], IT[ilev], Sc, NX[ilev], damp, epsi, ndt, R[ilev], ilev );
            end

            if ilev==1 iter_tot = iter_tot + iter; end

            if (ilev<nlev)
                B[ilev+1] .= 0.0;
                B[ilev+1] .= Restrict(R[ilev], B[ilev+1], XV[ilev], XV[ilev+1], 1);
            end
        end

        for ilev=nlev:-1:1
            #  damp    = 1-dmp/NX[ilev] # Turn off damping for MG

             if ilev == nlev && nlev>1
                 U[ilev] = DirectSolve( B[ilev], K[ilev], DX[ilev], NX[ilev] );
             else
                (U[ilev], Rd[ilev], R[ilev], iter) = PT_solver(U[ilev], B[ilev], DX[ilev], K[ilev], IT[ilev], Sc, NX[ilev], damp, epsi, ndt, R[ilev], ilev );
             end
            if ilev==1 iter_tot = iter_tot + iter; end
            if (ilev>1)      
                dU  = zeros(NX[ilev-1]);
                dU  .= Prolongate( dU, U[ilev], XV[ilev-1], XV[ilev],  1);
                U[ilev-1] .= U[ilev-1] + 1.0*vcoef*dU;
            end
        end
        u_mg = u_mg + U[1];
        q    = k.*diff(u_mg,dims=1)/dx;
        f    = b .- [0; diff(q,dims=1)/dx; 0];
        if (norm(f)/length(f) < 1*epsi) break; end
        @printf("it = %03d --- R = %2.2e\n", itMG, norm(f)/(length(f))); 
        B[1] .= f;
    end
    @printf("Converged in %02d iterations down to R = %2.2e\n", iter_tot, norm(RP)/(length(RP))); 

    p = plot()
    p = plot!(xv, u_dir, label="direct")
    p = plot!(xv, u_PT,  label="PT (damped)")
    p = plot!(xv, u_jac, label="Jacobi (damped)")
    p = plot!(xv, u_jac, label="GS Red-black (damped)")
    p = plot!(xv, u_jac, label="GMG (PT undamped)")
    display(p)

end
    
Poisson1D_MG_v2(50)


