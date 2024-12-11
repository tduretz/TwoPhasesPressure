using TwoPhasesPressure
using LinearAlgebra, ExtendableSparse, Printf
using Statistics
import Plots
using GLMakie, MathTeXEngine
Makie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))
using MAT

function main()
    # Adimensionnal numbers
    Ωl     = 10^-1         # Ratio √(k_ηf0 * (ηb + 4/3 * ηs)) / len
    Ωη     = 10^-1          # Ratio ηb / ηs
    Ωr     = 0.1            # Ratio inclusion radius / len
    Ωηi    = 10             # Ratio (inclusion viscosity) / (matrix viscosity)
    Ωp     = 1              # Ratio (ε̇bg * ηs) / P0
    # Independant
    ηs0    = 1              # Shear viscosity
    len    = 1              # Box size
    P0     = 1              # Initial ambiant pressure
    n      = 100            # Resolution
    # Dependant
    ηb0    = Ωη * ηs0       # Bulk viscosity
    k_ηf0  = (len.^2 * Ωl^2) / (ηb0 + 4/3 * ηs0) # Permeability / fluid viscosity
    r      = Ωr * len       # Inclusion radius
    ηs_inc = Ωηi * ηs0      # Inclusion shear viscosity
    ε̇bg    = Ωp * P0 / ηs0  # Background strain rate
    # Numerics
    xlim   = (min=-len/2, max=len/2)
    ylim   = (min=-len/2, max=len/2)
    nc     = (x=n, y=n)
    nv     = (x=nc.x+1, y=nc.y+1)
    nc     = (x=nc.x+0, y=nc.y+0)
    nv     = (x=nv.x+0, y=nv.y+0)
    Δ      = (x=(xlim.max-xlim.min)/nc.x, y=(ylim.max-ylim.min)/nc.y)
    x      = (c=LinRange(xlim.min-Δ.x/2, xlim.max+Δ.x/2, nc.x), v=LinRange(xlim.min-Δ.x, xlim.max+Δ.x, nv.x))
    y      = (c=LinRange(ylim.min-Δ.y/2, ylim.max+Δ.y/2, nc.y), v=LinRange(ylim.min-Δ.y, ylim.max+Δ.y, nv.y))
    # Primitive variables
    V      = (x=zeros(nv.x, nc.y+2), y=zeros(nc.x+2, nv.y))
    Pt     = zeros(nc...)
    Pf     = zeros(nc...)
    # Residuals
    Fm     = (x=zeros(nv.x, nc.y), y=zeros(nc.x, nv.y))
    FPt    = zeros(nc...)
    FPf    = zeros(nc...) 
    # Derived fields
    ε̇      = (xx=zeros(nc.x, nc.y), yy=zeros(nc.x, nc.y), xy=zeros(nv.x, nv.y))
    τ      = (xx=zeros(nc.x, nc.y), yy=zeros(nc.x, nc.y), xy=zeros(nv.x, nv.y))
    qD     = (x =zeros(nv.x, nc.y), y =zeros(nc.x, nv.y))
    divVs  = zeros(nc.x, nc.y)
    divqD  = zeros(nc.x, nc.y)
    # Materials
    k_ηf   = (x=k_ηf0*ones(nv.x, nc.y), y=k_ηf0*ones(nc.x, nv.y))
    ηs     = (c=ηs0*ones(nc...), v=ηs0*ones(nv...))
    ηb     = ηb0*ones(nc...)
    # Initial condition
    @. Pt  = P0
    @. Pf  = P0
    @. ηs.v[x.v^2 + (y.v.^2)'<r^2] = ηs_inc
    @. ηs.c = 0.25*(ηs.v[1:end-1,1:end-1] + ηs.v[2:end-0,1:end-1] + ηs.v[1:end-1,2:end-0] + ηs.v[2:end-0,2:end-0])
   
    # Pure shear
    BC     = (W=:Neumann, E=:Neumann, S=:Neumann, N=:Neumann)
    VxBC   = (S=zeros(nv.x), N=zeros(nv.x))
    VyBC   = (W=zeros(nv.y), E=zeros(nv.y))
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

    # Calculating mode
    nsigma = 2.8
    print(std(Pt[:]), " ", std(Pf[:]), " ")
    pt_sigma = sum(abs.(Pt[:] .- mean(Pt)) .> nsigma*std(Pt[:])) / length(Pt)
    pf_sigma = sum(abs.(Pf[:] .- mean(Pf)) .> nsigma*std(Pf[:])) / length(Pt)
    mode = (Int(pf_sigma == 0) - (Int(pf_sigma > 0)))  * (pt_sigma > 0)
    print(mode, " ", pt_sigma, " ", pf_sigma, "\n")

    # Visualisation
    ε̇xy_c = 0.25*(ε̇.xy[1:end-1,1:end-1] + ε̇.xy[2:end-0,1:end-1] + ε̇.xy[1:end-1,2:end-0] + ε̇.xy[2:end-0,2:end-0]) 
    ε̇II   = sqrt.( 0.5*ε̇.xx.^2 + 0.5*ε̇.yy.^2 + ε̇xy_c.^2 )
    τxy_c = 0.25*(τ.xy[1:end-1,1:end-1] + τ.xy[2:end-0,1:end-1] + τ.xy[1:end-1,2:end-0] + τ.xy[2:end-0,2:end-0]) 
    τII   = sqrt.( 0.5*τ.xx.^2 + 0.5*τ.yy.^2 + τxy_c.^2 )

    f1 = Figure(size = (1000, 800), fontsize=20)
    width_colorbar = 15
    CreateSubplot(f1, 1, 1, x.v, y.c, V.x[:,2:end-1], L"(A) Vx", width_colorbar)
    CreateSubplot(f1, 1, 2, x.c, y.v, V.y[2:end-1,:], L"(B) Vy", width_colorbar)
    CreateSubplot(f1, 1, 3, x.c, y.c, ε̇II, L"(C) \: ε̇II", width_colorbar)
    CreateSubplot(f1, 2, 1, x.c, y.c, Pt, L"(D) \: Pt", width_colorbar)
    CreateSubplot(f1, 2, 2, x.c, y.c, Pf, L"(E) \: Pf", width_colorbar)
    CreateSubplot(f1, 2, 3, x.c, y.c, τII, L"(F) \: τII", width_colorbar)
    CreateSubplot(f1, 3, 1, x.c, y.c, divVs, L"(G) \: \nabla{}Vs", width_colorbar)
    CreateSubplot(f1, 3, 2, x.c, y.c, divqD, L"(H) \: \nabla{}qD", width_colorbar)
    CreateSubplot(f1, 3, 3, x.c, y.c, ηs.c, L"(I) \: ηs", width_colorbar)
    DataInspector(f1) 
    display(f1)
    # save("mosaic_pureshear.png", f1, px_per_unit = 5)
    # save("mosaic_simpleshear.png", f1, px_per_unit = 5)
    # Saving data to .mat file, to be processed with matlab
    # file = matopen("nx500.mat", "w")
    # write(file, "xc", collect(x.c))
    # write(file, "xv", collect(x.v))
    # write(file, "yc", collect(y.c))
    # write(file, "yv", collect(y.v))
    # write(file, "Pt", Pt)
    # write(file, "Pf", Pf)
    # write(file, "Vx", V.x)
    # write(file, "Vy", V.y)
    # close(file)
end

# Function to create a subplot and colorbar
function CreateSubplot(figure, row, col, xdata, ydata, vdata, title, width_colorbar)
    ax = Axis(figure[row, 2*col - 1],
                title = title,
                xlabel = L"$x$ [m]",
                ylabel = L"$y$ [m]",
                aspect = DataAspect(),
                xticks = [-0.5, 0, 0.5])
    hm = GLMakie.heatmap!(ax, xdata, ydata, vdata, colormap = :thermal)
    Colorbar(figure[row, 2*col], hm, width = width_colorbar, ticklabelsize = 14 )
end

@time main()