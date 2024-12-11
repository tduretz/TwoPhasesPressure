
using MAT
using GLMakie, MathTeXEngine
Makie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

function main()
    vars = matread("systematicdata.mat")
    Ωl   = vars["length_number"]
    Ωη   = vars["eta_number"]
    mode = vars["mode"]

    f1 = Figure(size = (800, 800), fontsize=20)
    ax = f1[1, 1] = Axis(f1,
            title = "Pressure modes",
            xlabel = L"log_{10} \: Ωl",
            ylabel = L"log_{10} \: Ωη",
            aspect = DataAspect())
    heatmap!(ax, log10.(Ωl), log10.(Ωη), mode, colormap = :grays)
    DataInspector(f1) 
    display(f1)
    # save("systematics.png", f1, px_per_unit = 5)
    # save("systematics_lowres.png", f1, px_per_unit = 1)

end
@time main()
