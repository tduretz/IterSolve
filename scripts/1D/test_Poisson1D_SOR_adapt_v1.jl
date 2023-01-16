using Printf, Plots
import Statistics: mean
import LinearAlgebra: norm, dot
import SparseArrays: sparse, tril, triu, diag, spdiagm

#########################

function DirectSolve( b, k, dx, nvx )
    # Direct solver
    NumUx = 1:nvx;
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
    u_dir = M\b;
    return (u_dir,M)
end

#########################

function Poisson1D_DYSOR(n)

    # Domain
    nvx  = n*101
    ncx  = nvx-1
    xmin = -0.5
    xmax =  0.5
    dx   =  (xmax-xmin)/(ncx)
    # Forcing term (RHS)
    Amp  = 0.1
    Sig  = 0.05
    xc   = LinRange(xmin+dx/2.0, xmax-dx/2.0, ncx)
    xv   = LinRange(xmin, xmax, ncx+1)
    b    = zeros(nvx,1) .+ Amp*exp.(-(xv).^2 /Sig^2); 
    f    = zeros(nvx,1)
    q    = zeros(nvx-1)
    # Variable coefficient
    k    = ones(ncx,1) .+ 0.0*Amp*exp.(-(xc).^2 /Sig^2); 
    # Initial solution
    u    = zeros(nvx,1);

    #-------------------------------------------------------------------------%

    # Residual
    q   .= .-k.*diff(u,dims=1)/dx;
    f   .= b .+ [0; diff(q,dims=1)./dx; 0];
    f0   = copy(f)
    @printf( "Initial residual: %2.2e\n", norm(f)/((length(f))));

    #-------------------------------------------------------------------------%

    # Direct Solve
    @printf("\n*** Direct Solve (UMFPACK) ***\n");
    (u_dir, K) = DirectSolve( b, k, dx, nvx );
    q   .= .-k.*diff(u_dir,dims=1)/dx;
    f   .= b .- [0; diff(q,dims=1)./dx; 0];
    @printf("Converged down to R = %2.2e\n", norm(f)/(length(f))); 

    #-------------------------------------------------------------------------%
    
    # SOR
    @printf("\n*** SOR ***\n");
    u_SOR = zeros(size(u_dir))
    L     = .-tril(K,-1)
    U     = .-triu(K, 1)
    D     = spdiagm(diag(K))
    # ω = 4*(0.5-1/ncx)
    ω     = 1.971
    tol   = 1e-10
    itmax = 1000
    # SOR Solve with optimal ω
    its = 0
    for it=1:itmax
        its   += 1
        u_SOR .= (D .- ω.*L) \ (ω.*b .+ (1.0.-ω).*(D*u_SOR) .+ ω.*(U*u_SOR) )
        q     .= .-k.*diff(u_SOR,dims=1)/dx;
        f     .= b .- [0; diff(q,dims=1)./dx; 0];
        if norm(f)/(length(f))<tol break end 
    end
    @printf("  SOR converged in %04d\n", its)
    @printf("Converged down to R = %2.2e\n", norm(f)/(length(f))); 

    #-------------------------------------------------------------------------%
    
    # DYSOR
    u_SOR .= 0.0
    ωv     = LinRange(0.0, 2.0, 20)
    f0_vec = zeros(size(ωv))
    f0_log = zeros(itmax)
    ω_log  = zeros(itmax)
    ω      = 1.0 # let's start with ω = 1.0
    its    = 0
    for it=1:itmax
        its   += 1
        u_SOR .= (D .- ω.*L) \ (ω.*b .+ (1.0.-ω).*(D*u_SOR) .+ ω.*(U*u_SOR) )
        q     .= .-k.*diff(u_SOR,dims=1)/dx;
        f     .= b .- [0; diff(q,dims=1)./dx; 0];
        if norm(f)/(length(f))<tol break end 

        # Optimal ω search after Liu (2021) 
        a1 = D*u_SOR
        a2 = -L*u_SOR
        b1 = D*u_SOR
        b2 = b .- D*u_SOR .+ U*u_SOR
        ωv = LinRange(1.0,2.1,20)
        f0_vec .= 0.0
        for ils in eachindex(ωv)
            ω1 = ωv[ils]
            f1 = dot(a1.+ω1.*a2, a1.+ω1.*a2)
            f2 = dot(b1.+ω1.*b2, b1.+ω1.*b2)
            f3 = dot(a1.+ω1.*a2, b1.+ω1.*b2)
            f0 = f1*f2/f3^2
            f0_vec[ils] = f0
        end
        (val,ind)   = findmin(f0_vec)
        ω           = ωv[ind]
        ω_log[its]  = ωv[ind]
        f0_log[its] = f0_vec[ind]
    end
    @printf("DYSOR converged in %04d\n", its)
    @printf("Converged down to R = %2.2e\n", norm(f)/(length(f))); 

    #-------------------------------------------------------------------------%
    p1 = plot(ylabel="u", xlabel="x")
    p1 = plot!(xv, u_dir, label="direct")
    p1 = plot!(xv, u_SOR, label="SOR")
    p2 = plot(1:its,  ω_log[1:its], ylabel="ω" , xlabel="its", label=:none)
    p3 = plot(1:its, f0_log[1:its], ylabel="f0", xlabel="its", label=:none)
    display( plot(p1, p2, p3, layout=(3,1)))
end
    
Poisson1D_DYSOR(2)
