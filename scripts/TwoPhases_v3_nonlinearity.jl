using TwoPhasesPressure
using LinearAlgebra, ExtendableSparse, Printf
import Statistics: mean
import Plots

function main()

    # permeabilité de reference (k_ηf_ref), porosité fixée
    # k_ηf0 = f(Ωl, nϕ, k_ηf_ref)

    # Dimensionaless numbers
    Ωl     = 10^0      # Ratio √(k_ηf0 * (ηb + 4/3 * ηs)) / len
    Ωη     = 10^0      # Ratio ηb / ηs
    ηs_ηs0 = 10.0       # Ratio (inclusion viscosity) / (matrix viscosity)
    # Independent
    ηs0    = 1.0        # Shear viscosity
    len    = 1.0       # Box size
    ε̇bg    = 1          # Background strain rate
    ϕ0     = 0.01
    ϕref   = 0.01       # Reference porosity for which k_ηf_0 is a reference permeability
    nϕ     = 3.0
    ρs0    = 3000
    βs     = 1e-6
    dt     = 1e-6
    # Dependent
    ηb0    = Ωη * ηs0   # Bulk viscosity
    k_ηf0  = (len.^2 * Ωl^2) / (ηb0 + 4/3 * ηs0) / ϕref^nϕ   # Permeability / fluid viscosity 
    k_ηf_ref = k_ηf0*ϕ0^nϕ
    r      = len/10.0   # Inclusion radius

    xlim = (min=-len/2, max=len/2)
    ylim = (min=-len/2, max=len/2)
    nc   = (x=50, y=50)
    nv   = (x=nc.x+1, y=nc.y+1)
    Δ    = (x=(xlim.max-xlim.min)/nc.x, y=(ylim.max-ylim.min)/nc.y)
    x    = (c=LinRange(xlim.min-Δ.x/2, xlim.max+Δ.x/2, nc.x), v=LinRange(xlim.min-Δ.x, xlim.max+Δ.x, nv.x))
    y    = (c=LinRange(ylim.min-Δ.y/2, ylim.max+Δ.y/2, nc.y), v=LinRange(ylim.min-Δ.y, ylim.max+Δ.y, nv.y))

    # Primitive variables
    V     = (x=zeros(nv.x, nc.y+2), y=zeros(nc.x+2, nv.y))
    Pt    =  ones(nc...)
    Pf    = zeros(nc...)
    # Residuals
    ϵ     = 1e-11
    Fm    = (x=zeros(nv.x, nc.y), y=zeros(nc.x, nv.y))
    FPt   = zeros(nc...)
    FPf   = zeros(nc...) 
    # Derived fields
    ε̇     = (xx=zeros(nc.x, nc.y), yy=zeros(nc.x, nc.y), xy=zeros(nv.x, nv.y))
    τ     = (xx=zeros(nc.x, nc.y), yy=zeros(nc.x, nc.y), xy=zeros(nv.x, nv.y))
    qD    = (x =zeros(nv.x, nc.y), y =zeros(nc.x, nv.y))
    divVs = zeros(nc.x, nc.y)
    divqD = zeros(nc.x, nc.y)
    # Materials
    k_ηf  = (x=k_ηf0*ones(nv.x, nc.y), y=k_ηf0*ones(nc.x, nv.y))
    ηs    = (c=ηs0*ones(nc...), v=ηs0*ones(nv...))
    ηb    = ηb0*ones(nc...)
    ηϕ    = ηb0*ones(nc...)
    ϕ     = (c=zeros(nc.x, nc.y), x=zeros(nv.x-2, nc.y), y =zeros(nc.x, nv.y-2))
    ϕold  = zeros(nc...)
    ρsold  = zeros(nc...)
    ρs    = ρs0 * ones(nc...)
    
    # Initial condition
    @. ηs.v[x.v^2 + (y.v.^2)'<r^2] = ηs_ηs0 .* ηs0
    for smo=1:round(nc.x^2 / 1000)
        Ii              = 2:nv.x-1;
        kdiff           = 0.1;
        @. ηs.v[Ii,:]       = ηs.v[Ii,:] + kdiff * (ηs.v[Ii+1,:] - 2*ηs.v[Ii,:] + ηs.v[Ii-1,:]);
        @. ηs.v[:,Ii]       = ηs.v[:,Ii] + kdiff * (ηs.v[:,Ii+1] - 2*ηs.v[:,Ii] + ηs.v[:,Ii-1]);
    end
    @. ηs.c = 0.25*(ηs.v[1:end-1,1:end-1] + ηs.v[2:end-0,1:end-1] + ηs.v[1:end-1,2:end-0] + ηs.v[2:end-0,2:end-0])
    @. ϕ.c   = ϕ0
    ηs_ini = (c = copy(ηs.c), v = copy(ηs.v))
    ηb_ini = copy(ηb)

    # Pure shear
    BC   = (W=:Neumann, E=:Neumann, S=:Neumann, N=:Neumann)
    VxBC = (S=zeros(nv.x), N=zeros(nv.x))
    VyBC = (W=zeros(nv.y), E=zeros(nv.y))
    @. V.x[:,2:end-1]  = x.v*ε̇bg - 0*y.c' 
    @. V.y[2:end-1,:]  = x.c*0   - ε̇bg*y.v'

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
    F       = zeros(maximum(Num.Pf))
    nF, nF0 = zeros(4), zeros(4)

    # Time loop
    time = 0
    for t = 1:10
        @. ϕold  = copy(ϕ.c)
        @. ρsold = copy(ρs)
        for it = 1:100
            ResidualsNonLinear!(Fm, FPt, FPf, V, Pt, Pf, divVs, divqD, ε̇, τ, qD, ηs, ηb, ηϕ, k_ηf, Δ, BC, VxBC, VyBC, ϕ, ϕold, k_ηf0, nϕ, dt, ηs_ini, ηb_ini, βs, ρs0, ρs, ρsold  )
            nF .= [norm(Fm.x)/length(Fm.x); norm(Fm.y)/length(Fm.y); norm(FPt)/length(FPt); norm(FPf)/length(FPf)]
            if it==1 nF0 .= nF end
            rel_tol = maximum(nF     ) < ϵ
            abs_tol = maximum(nF./nF0) < ϵ
            if (abs_tol || rel_tol)
                # print("Time step ", t, " converged in ", it, " iterations\n")
                # @printf("Fmx: abs = %1.4e --- rel = %1.4e \n", nF[1], nF[1]./nF0[1])
                # @printf("Fmy: abs = %1.4e --- rel = %1.4e \n", nF[2], nF[2]./nF0[2])
                # @printf("Fpt: abs = %1.4e --- rel = %1.4e \n", nF[3], nF[3]./nF0[3])
                # @printf("Fpf: abs = %1.4e --- rel = %1.4e \n", nF[4], nF[4]./nF0[4]) 
                break
            elseif it == 100
                print("Time step ", t, " did not converge\n")
            end
            F[Num.Vx] = Fm.x[:]
            F[Num.Vy] = Fm.y[:]
            F[Num.Pt] = FPt[:]
            F[Num.Pf] = FPf[:]
            # Assembly of linear system
            K = Assembly( ηs, ηϕ, k_ηf, BC, Num, nv, nc, Δ )
            # Solution of linear system
            δx = -K\F
            # Correct solution fields
            V.x[:,2:end-1] .+= δx[Num.Vx]
            V.y[2:end-1,:] .+= δx[Num.Vy]
            Pt             .+= δx[Num.Pt]
            Pf             .+= δx[Num.Pf]
        end # End iteration loop
    if t%50 == 0 print("Ended time step ", t, "\n") end
    time += dt
    end # End time loop
    @show mean(Pt)
    @show mean(Pf)
    @show minimum(ηs.c)
    @show minimum(ϕ.c)
    # Final residuals
    ResidualsNonLinear!(Fm, FPt, FPf, V, Pt, Pf, divVs, divqD, ε̇, τ, qD, ηs, ηb, ηϕ, k_ηf, Δ, BC, VxBC, VyBC, ϕ, ϕold, k_ηf0, nϕ, dt, ηs_ini, ηb_ini, βs, ρs0, ρs, ρsold  )

    # Visualisation
    ε̇xy_c = 0.25*(ε̇.xy[1:end-1,1:end-1] + ε̇.xy[2:end-0,1:end-1] + ε̇.xy[1:end-1,2:end-0] + ε̇.xy[2:end-0,2:end-0]) 
    ε̇II   = sqrt.( 0.5*ε̇.xx.^2 + 0.5*ε̇.yy.^2 + ε̇xy_c.^2 )
    τxy_c = 0.25*(τ.xy[1:end-1,1:end-1] + τ.xy[2:end-0,1:end-1] + τ.xy[1:end-1,2:end-0] + τ.xy[2:end-0,2:end-0]) 
    τII   = sqrt.( 0.5*τ.xx.^2 + 0.5*τ.yy.^2 + τxy_c.^2 )
    p1 = Plots.heatmap(x.v, y.c, V.x[:,2:end-1]', title="Vx")
    p2 = Plots.heatmap(x.c, y.v, V.y[2:end-1,:]', title="Vy")
    p3 = Plots.heatmap(x.c, y.c, ηs.c',            title="ηs")
    p4 = Plots.heatmap(x.c, y.c, Pt',              title="Pt")
    p5 = Plots.heatmap(x.c, y.c, Pf',             title="Pf")
    p6 = Plots.heatmap(x.c, y.c, ηb',            title="ηb")
    p7 = Plots.heatmap(x.c, y.c, divVs',          title="divVs")
    p8 = Plots.heatmap(x.c, y.c, ρs',          title="ρs")
    p9 = Plots.heatmap(x.c, y.c, ϕ.c * 100',            title="ϕ %")
    display(Plots.plot(p1, p2, p3, p4, p5, p6, p7, p8, p9))
end

@time main()