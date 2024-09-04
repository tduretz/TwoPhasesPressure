using TwoPhasesPressure
using LinearAlgebra, ExtendableSparse, Printf
import Statistics: mean
using GLMakie, MathTeXEngine
Makie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

function main()
    # Dimensionaless numbers
    Ωl     = 10^-1      # Ratio √(k_ηf0 * (ηb + 4/3 * ηs)) / len
    Ωη     = 10^-1      # Ratio ηb / ηs
    Ωr     = 0.1        # Ratio inclusion radius / len
    Ωηi    = 10         # Ratio (inclusion viscosity) / (matrix viscosity)
    Ωp     = 1          # Ratio (ε̇bg * ηs) / P0    
    # Independent
    ηs0    = 1.0        # Shear viscosity
    len    = 1.0        # Box size
    P0     = 1          # Initial ambiant pressure
    ϕ0     = 0.01
    ϕref   = 0.01       # Reference porosity for which k_ηf_0 is a reference permeability
    nϕ     = 3.0
    ρs0    = 1
    βs     = 1e-9
    dt     = 1e-4
    # Dependent
    ηb0    = Ωη * ηs0   # Bulk viscosity
    k_ηf0  = (len.^2 * Ωl^2) / (ηb0 + 4/3 * ηs0) / ϕref^nϕ   # Permeability / fluid viscosity 
    k_ηf_ref = k_ηf0*ϕ0^nϕ
    r      = Ωr * len       # Inclusion radius
    ηs_inc = Ωηi * ηs0      # Inclusion shear viscosity
    ε̇bg    = Ωp * P0 / ηs0  # Background strain rate

    xlim = (min=-len/2, max=len/2)
    ylim = (min=-len/2, max=len/2)
    nc   = (x=100, y=100)
    nv   = (x=nc.x+1, y=nc.y+1)
    Δ    = (x=(xlim.max-xlim.min)/nc.x, y=(ylim.max-ylim.min)/nc.y)
    x    = (c=LinRange(xlim.min-Δ.x/2, xlim.max+Δ.x/2, nc.x), v=LinRange(xlim.min-Δ.x, xlim.max+Δ.x, nv.x))
    y    = (c=LinRange(ylim.min-Δ.y/2, ylim.max+Δ.y/2, nc.y), v=LinRange(ylim.min-Δ.y, ylim.max+Δ.y, nv.y))

    # Primitive variables
    V     = (x=zeros(nv.x, nc.y+2), y=zeros(nc.x+2, nv.y))
    Pt    = zeros(nc...)
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
    @. Pt  = P0
    @. Pf  = P0
    @. ηs.v[x.v^2 + (y.v.^2)'<r^2] = ηs_inc
    @. ηs.c = 0.25*(ηs.v[1:end-1,1:end-1] + ηs.v[2:end-0,1:end-1] + ηs.v[1:end-1,2:end-0] + ηs.v[2:end-0,2:end-0])
    @. ϕ.c   = ϕ0
    ηs_ini = (c = copy(ηs.c), v = copy(ηs.v))
    ηb_ini = copy(ηb)
    ϕ_ini  = copy(ϕ.c)

    # Pure shear
    BC   = (W=:Neumann, E=:Neumann, S=:Neumann, N=:Neumann)
    VxBC = (S=zeros(nv.x), N=zeros(nv.x))
    VyBC = (W=zeros(nv.y), E=zeros(nv.y))
    @. V.x[:,2:end-1]  = x.v*ε̇bg - 0*y.c' 
    @. V.y[2:end-1,:]  = x.c*0   - ε̇bg*y.v'

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
            ResidualsNonLinear!(Fm, FPt, FPf, V, Pt, Pf, divVs, divqD, ε̇, τ, qD, ηs, ηb, ηϕ, k_ηf, Δ, BC, VxBC, VyBC, ϕ, ϕ0, ϕold, k_ηf0, nϕ, dt, ηs_ini, ηb_ini, βs, ρs0, ρs, ρsold, P0  )
            nF .= [norm(Fm.x)/length(Fm.x); norm(Fm.y)/length(Fm.y); norm(FPt)/length(FPt); norm(FPf)/length(FPf)]
            if it==1 nF0 .= nF end
            rel_tol = maximum(nF     ) < ϵ
            abs_tol = maximum(nF./nF0) < ϵ
            if (abs_tol || rel_tol)
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
    print("Ended time loop ", t, "\n")
    time += dt
    end # End time loop
    # Final residuals
    ResidualsNonLinear!(Fm, FPt, FPf, V, Pt, Pf, divVs, divqD, ε̇, τ, qD, ηs, ηb, ηϕ, k_ηf, Δ, BC, VxBC, VyBC, ϕ, ϕ0, ϕold, k_ηf0, nϕ, dt, ηs_ini, ηb_ini, βs, ρs0, ρs, ρsold, P0  )

    f1 = Figure(size = (1000, 800), fontsize=20)
    width_colorbar = 15
    CreateSubplot(f1, 1, 1, x.v, y.c, V.x[:,2:end-1], L"(A) Vx", width_colorbar)
    CreateSubplot(f1, 1, 2, x.c, y.v, V.y[2:end-1,:], L"(B) Vy", width_colorbar)
    CreateSubplot(f1, 1, 3, x.c, y.c, divVs, L"(C) \: \nabla{}Vs", width_colorbar)
    CreateSubplot(f1, 2, 1, x.c, y.c, Pt, L"(D) \: Pt", width_colorbar)
    CreateSubplot(f1, 2, 2, x.c, y.c, Pf, L"(E) \: Pf", width_colorbar)
    CreateSubplot(f1, 2, 3, x.c, y.c, ρs .- ρs0, L"(F) \: ρs - ρs_0", width_colorbar)
    CreateSubplot(f1, 3, 1, x.c, y.c, ηs.c ./ ηs_ini.c, L"(G) \: ηs / ηs_0", width_colorbar)
    CreateSubplot(f1, 3, 2, x.c, y.c, ηb ./ ηb_ini, L"(H) \: ηb / ηb_0", width_colorbar)
    CreateSubplot(f1, 3, 3, x.c, y.c, ϕ.c ./ ϕ_ini, L"(I) \: ϕ / ϕ_0", width_colorbar)
    DataInspector(f1) 
    display(f1)
    # save("nonlinearity.png", f1, px_per_unit = 5)

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