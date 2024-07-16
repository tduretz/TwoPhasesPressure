using TwoPhasesPressure
using LinearAlgebra, ExtendableSparse, Printf
using Statistics
import Plots

# using GLMakie, MathTeXEngine
# Makie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

function main()
    # Adimensionnal numbers
    ηs_ηs0 = 10         # Ratio (inclusion viscosity) / (matrix viscosity)
    # Independant
    ηs0    = 1          # Shear viscosity
    len    = 1          # Box size
    ε̇bg    = 1          # Background strain rate

    xlim = (min=-len/2, max=len/2)
    ylim = (min=-len/2, max=len/2)
    nc   = (x=100, y=100)
    nv   = (x=nc.x+1, y=nc.y+1)
    nc   = (x=nc.x+0, y=nc.y+0)
    nv   = (x=nv.x+0, y=nv.y+0)
    Δ    = (x=(xlim.max-xlim.min)/nc.x, y=(ylim.max-ylim.min)/nc.y)
    x    = (c=LinRange(xlim.min-Δ.x/2, xlim.max+Δ.x/2, nc.x), v=LinRange(xlim.min-Δ.x, xlim.max+Δ.x, nv.x))
    y    = (c=LinRange(ylim.min-Δ.y/2, ylim.max+Δ.y/2, nc.y), v=LinRange(ylim.min-Δ.y, ylim.max+Δ.y, nv.y))

    # Loop over Ωη and Ωl
    Ωl_loop = 10.0 .^(-2:0.5:1)
    Ωη_loop = 10.0 .^(-4:0.5:0)
    mode = zeros(length(Ωl_loop), length(Ωη_loop))
    for i = eachindex(Ωl_loop), j = eachindex(Ωη_loop)
        Ωl = Ωl_loop[i]
        Ωη = Ωη_loop[j]
        # Dependant
        ηb0    = Ωη * ηs0   # Bulk viscosity
        k_ηf0  = (len.^2 * Ωl^2) / (ηb0 + 4/3 * ηs0) # Permeability / fluid viscosity
        r      = len/10     # Inclusion radius
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
        @. ηs.v[x.v^2 + (y.v.^2)'<r^2] = ηs_ηs0 .* ηs0
        @. ηs.c = 0.25*(ηs.v[1:end-1,1:end-1] + ηs.v[2:end-0,1:end-1] + ηs.v[1:end-1,2:end-0] + ηs.v[2:end-0,2:end-0])
    
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
        nsigma = 3.
        pt_sigma = sum(abs.(Pt[:] .- mean(Pt)) .> nsigma*std(Pt[:])) / length(Pt)
        pf_sigma = sum(abs.(Pf[:] .- mean(Pf)) .> nsigma*std(Pf[:])) / length(Pt)
        mode[i,j] = (Int(pf_sigma == 0) - (Int(pf_sigma > 0)))  * (pt_sigma > 0)
    end
    p1 = Plots.heatmap(log10.(Ωl_loop), log10.(Ωη_loop), mode', xlabel="log Ωl", ylabel="log Ωη")
    display(Plots.plot(p1))
end

@time main()