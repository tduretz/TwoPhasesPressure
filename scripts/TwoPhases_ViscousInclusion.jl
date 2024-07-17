using TwoPhasesPressure
using LinearAlgebra, ExtendableSparse, Printf, ExactFieldSolutions

import Plots

function main()

    ηs0   = 1.0
    k_ηf0 = 100.0
    ηb0   = 1000.

    xlim = (min=-0.5, max=0.5)
    ylim = (min=-0.5, max=0.5)
    nc   = (x=200, y=200)
    nv   = (x=nc.x+1, y=nc.y+1)
    nc   = (x=nc.x+0, y=nc.y+0)
    nv   = (x=nv.x+0, y=nv.y+0)
    Δ    = (x=(xlim.max-xlim.min)/nc.x, y=(ylim.max-ylim.min)/nc.y)
    x    = (c=LinRange(xlim.min-Δ.x/2, xlim.max+Δ.x/2, nc.x), v=LinRange(xlim.min-Δ.x, xlim.max+Δ.x, nv.x))
    y    = (c=LinRange(ylim.min-Δ.y/2, ylim.max+Δ.y/2, nc.y), v=LinRange(ylim.min-Δ.y, ylim.max+Δ.y, nv.y))

    # Primitive variables
    V     = (x=zeros(nv.x, nc.y+2), y=zeros(nc.x+2, nv.y))
    Pt    = zeros(nc...)
    Pf    = zeros(nc...)
    # Residuals
    Fm    = (x=zeros(nv.x, nc.y), y=zeros(nc.x, nv.y))
    FPt   = zeros(nc...)
    FPf   = zeros(nc...) 
    # Derived fields
    ε̇     = (xx=zeros(nc.x, nc.y), yy=zeros(nc.x, nc.y), xy=zeros(nv.x, nv.y))
    τ     = (xx=zeros(nc.x, nc.y), yy=zeros(nc.x, nc.y), xy=zeros(nv.x, nv.y))
    qD    = (x =zeros(nv.x, nc.y), y =zeros(nc.x, nv.y))
    ϕ     = zeros(nc.x, nc.y)
    divVs = zeros(nc.x, nc.y)
    divqD = zeros(nc.x, nc.y)
    # Materials
    k_ηf = (x=k_ηf0*ones(nv.x, nc.y), y=k_ηf0*ones(nc.x, nv.y))
    ηs   = (c=ηs0*ones(nc...), v=ηs0*ones(nv...))
    ηb   = ηb0*ones(nc...)

    # Initial condition
    r   = 0.2
    @. ηs.v[x.v^2 + (y.v.^2)'<r^2] = 100.
    @. ηs.c = 0.25*(ηs.v[1:end-1,1:end-1] + ηs.v[2:end-0,1:end-1] + ηs.v[1:end-1,2:end-0] + ηs.v[2:end-0,2:end-0])

    # Boundary condition
    ε̇bg = 1.

    # Pure shear
    BC   = (W=:Dirichlet, E=:Dirichlet, S=:Dirichlet, N=:Dirichlet)
    VxBC = (S=zeros(nv.x), N=zeros(nv.x))
    VyBC = (W=zeros(nv.y), E=zeros(nv.y))

    for i=1:nv.x
        # South
        X = [x.v[i], ylim.min]
        s = Stokes2D_Schmid2003( X )
        VxBC.S[i] = s.V[1]
        # North
        X = [x.v[i], ylim.max]
        s = Stokes2D_Schmid2003( X )
        VxBC.N[i] = s.V[1]
    end

    for j=1:nv.y
        # West
        X = [xlim.min, y.v[j]]
        s = Stokes2D_Schmid2003( X )
        VyBC.W[j] = s.V[2]
        # East
        X = [xlim.max, y.v[j]]
        s = Stokes2D_Schmid2003( X )
        VyBC.E[j] = s.V[2]
    end

    for i=1:nv.x, j=1:nc.y
        X = [x.v[i], y.c[j]]
        s = Stokes2D_Schmid2003( X )
        V.x[i,j+1] = s.V[1]
    end
    for i=1:nc.x, j=1:nv.y
        X = [x.c[i], y.v[j]]
        s = Stokes2D_Schmid2003( X )
        V.y[i+1,j] = s.V[2]
    end
    for i=1:nc.x, j=1:nc.y
        X = [x.c[i], y.c[j]]
        s = Stokes2D_Schmid2003( X )
        Pt[i,j] = s.p
    end
    V_ana = (x=copy(V.x), y=copy(V.y))
    P_ana = copy(Pt)



    # # Simple shear
    # BC   = (W=:Neumann, E=:Neumann, S=:Dirichlet, N=:Dirichlet)
    # VxBC = (S=ε̇bg*ylim.min*ones(nv.x), N=ε̇bg*ylim.max*ones(nv.x))
    # VyBC = (W=zeros(nv.y), E=zeros(nv.y))
    # @. V.x[:,2:end-1]  = 0*x.v + ε̇bg*y.c' 
    # @. V.y[2:end-1,:]  = x.c*0 - 0*ε̇bg*y.v'

    # Numbering
    off    = [nv.x*nc.y, nv.x*nc.y+nc.x*nv.y, nv.x*nc.y+nc.x*nv.y+nc.x*nc.y, nv.x*nc.y+nc.x*nv.y+2*nc.x*nc.y]
    Num    = (Vx=reshape(1:nv.x*nc.y, nv.x, nc.y) , Vy=reshape(off[1]+1:off[1]+nc.x*nv.y, nc.x, nv.y), 
              Pt=reshape(off[2]+1:off[2]+nc.x*nc.y,nc...), Pf=reshape(off[3]+1:off[3]+nc.x*nc.y,nc...) )

    # Initial residuals
    F      = zeros(maximum(Num.Pf))
    Residuals!(Fm, FPt, FPf, V, Pt, Pf, divVs, divqD, ε̇, τ, qD, ηs, ηb, k_ηf, Δ, BC, VxBC, VyBC  )    
    
    F[Num.Vx] = Fm.x[:]
    F[Num.Vy] = Fm.y[:]
    F[Num.Pt] = FPt[:]
    F[Num.Pf] = FPf[:]

    # Assembly of linear system
    K = Assembly( ηs, ηb, k_ηf, BC, Num, nv, nc, Δ )
    
    # Solution of linear system
    δx = -K\F

    # Extract correction to solution fields
    V.x[:,2:end-1] .+= δx[Num.Vx]
    V.y[2:end-1,:] .+= δx[Num.Vy]
    Pt             .+= δx[Num.Pt]
    Pf             .+= δx[Num.Pf]

    # Final residuals
    Residuals!(Fm, FPt, FPf, V, Pt, Pf, divVs, divqD, ε̇, τ, qD, ηs, ηb, k_ηf, Δ, BC, VxBC, VyBC )

    # Visualisation
    p1 = Plots.heatmap(x.c, y.c, Pt', title = "Numerics")
    p2 = Plots.heatmap(x.c, y.c, P_ana', title = "Analytics")
    p3 = Plots.heatmap(x.c, y.c, Pt'.-P_ana', title = "Difference")
    display(Plots.plot(p1, p2, p3))

    @printf("Pressure error = %1.4e\n", norm(Pt.-P_ana)/length(Pt) )

end

@time main()