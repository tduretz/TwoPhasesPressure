using TwoPhasesPressure
using LinearAlgebra, ExtendableSparse, Printf

import Plots

# using GLMakie, MathTeXEngine
# Makie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

function main()
    # Adimensionnal numbers
    Ωl     = 10^0        # Ratio √(k_ηf0 * (ηb + 4/3 * ηs)) / len
    Ωη     = 10^-4       # Ratio ηb / ηs
    ηs_ηs0 = 10         # Ratio (inclusion viscosity) / (matrix viscosity)
    # Independant
    ηs0    = 1          # Shear viscosity
    len    = 1          # Box size
    ε̇bg    = 1          # Background strain rate
    ϕ0     = 0.01
    # Dependant
    ηb0    = Ωη * ηs0   # Bulk viscosity
    k_ηf0  = (len.^2 * Ωl^2) / (ηb0 + 4/3 * ηs0) # Permeability / fluid viscosity
    r      = len/10     # Inclusion radius

    xlim = (min=-len/2, max=len/2)
    ylim = (min=-len/2, max=len/2)
    nc   = (x=100, y=100)
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
    
    divVs = zeros(nc.x, nc.y)
    divqD = zeros(nc.x, nc.y)
    # Materials
    k_ηf = (x=k_ηf0*ones(nv.x, nc.y), y=k_ηf0*ones(nc.x, nv.y))
    ηs   = (c=ηs0*ones(nc...), v=ηs0*ones(nv...))
    ηb   = ηb0*ones(nc...)
    ϕ     = (c=zeros(nc.x, nc.y), x=zeros(nv.x-2, nc.y), y =zeros(nc.x, nv.y-2))

    # Initial condition
    @. ηs.v[x.v^2 + (y.v.^2)'<r^2] = ηs_ηs0 .* ηs0
    @. ηs.c = 0.25*(ηs.v[1:end-1,1:end-1] + ηs.v[2:end-0,1:end-1] + ηs.v[1:end-1,2:end-0] + ηs.v[2:end-0,2:end-0])
    @. ϕ.c   = ϕ0
    
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
    F      = zeros(maximum(Num.Pf))

    # Time loop
    dt = 1e-4
    ϕold = copy(ϕ.c)
    for it = 1:10
        
        # Residuals!(Fm, FPt, FPf, V, Pt, Pf, divVs, divqD, ε̇, τ, qD, ηs, ηb, k_ηf, Δ, BC, VxBC, VyBC)    
        ResidualsNonLinear!(Fm, FPt, FPf, V, Pt, Pf, divVs, divqD, ε̇, τ, qD, ηs, ηb, k_ηf, Δ, BC, VxBC, VyBC, ϕ, ϕold, dt  )    
        
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
    end # End time loop

    # Final residuals
    ResidualsNonLinear!(Fm, FPt, FPf, V, Pt, Pf, divVs, divqD, ε̇, τ, qD, ηs, ηb, k_ηf, Δ, BC, VxBC, VyBC, ϕ, ϕold, dt  )    

    # Visualisation
    ε̇xy_c = 0.25*(ε̇.xy[1:end-1,1:end-1] + ε̇.xy[2:end-0,1:end-1] + ε̇.xy[1:end-1,2:end-0] + ε̇.xy[2:end-0,2:end-0]) 
    ε̇II   = sqrt.( 0.5*ε̇.xx.^2 + 0.5*ε̇.yy.^2 + ε̇xy_c.^2 )
    τxy_c = 0.25*(τ.xy[1:end-1,1:end-1] + τ.xy[2:end-0,1:end-1] + τ.xy[1:end-1,2:end-0] + τ.xy[2:end-0,2:end-0]) 
    τII   = sqrt.( 0.5*τ.xx.^2 + 0.5*τ.yy.^2 + τxy_c.^2 )
    lc    = sqrt.(k_ηf0 * (ηb0 + 4/3 * ηs0))
    println("Compaction length ratio: ",lc/len)
    println("ηs0 / ηb0: ", ηs0 / ηb0)
    p1 = Plots.heatmap(x.v, y.c, V.x[:,2:end-1]', title="Vx")
    p2 = Plots.heatmap(x.c, y.v, V.y[2:end-1,:]', title="Vy")
    p3 = Plots.heatmap(x.c, y.c, ε̇II', title="ε̇II")
    p4 = Plots.heatmap(x.c, y.c, Pt', title="Pt")
    p5 = Plots.heatmap(x.c, y.c, Pf', title="Pf")
    p6 = Plots.heatmap(x.c, y.c, τII', title="τII")
    p7 = Plots.heatmap(x.c, y.c, divVs', title="divVs")
    p8 = Plots.heatmap(x.c, y.c, divqD', title="divqD")
    p9 = Plots.heatmap(x.c, y.c, ϕ.c', title="ϕ")
    display(Plots.plot(p1, p2, p3, p4, p5, p6, p7, p8, p9))

    # f   = Figure(size = (500, 500), fontsize=25)
    # ax1 = Axis(f[1, 1], title = L"P", xlabel = L"$x$ [m]", ylabel = L"$y$ [m]")
    # hm  = heatmap!(ax1, x.c, y.c, Pt', colormap = :turbo)
    # colsize!(f.layout, 1, Aspect(1, 1.0))
    # Colorbar(f[1, 2], hm, label = "Phases", width = 20, labelsize = 25, ticklabelsize = 14 )
    # DataInspector(f) 
    # display(f)

end

@time main()