function Residuals!(Fm, FPt, FPf, V, Pt, Pf, divVs, divqD, ε̇, τ, qD, ηs, ηb, k_ηf, Δ, BC, VxBC, VyBC  )
    @. V.x[:,1]   = (BC.S==:Neumann)*V.x[:,2]     + (BC.S==:Dirichlet)*(2*VxBC.S - V.x[:,2])
    @. V.x[:,end] = (BC.N==:Neumann)*V.x[:,end-1] + (BC.N==:Dirichlet)*(2*VxBC.N - V.x[:,end-1])
    @. V.y[1,:]   = (BC.W==:Neumann)*V.y[2,:]     + (BC.W==:Dirichlet)*(2*VyBC.W - V.y[2,:])
    @. V.y[end,:] = (BC.E==:Neumann)*V.y[end-1,:] + (BC.E==:Dirichlet)*(2*VyBC.E - V.y[end-1,:])
    @. divVs = (V.x[2:end,2:end-1] - V.x[1:end-1,2:end-1])/Δ.x + (V.y[2:end-1,2:end] - V.y[2:end-1,1:end-1])/Δ.y
    @. ε̇.xx  = (V.x[2:end,2:end-1] - V.x[1:end-1,2:end-1])/Δ.x - 1.0/3.0*divVs
    @. ε̇.yy  = (V.y[2:end-1,2:end] - V.y[2:end-1,1:end-1])/Δ.y - 1.0/3.0*divVs
    @. ε̇.xy  = 0.5*( (V.x[:,2:end] - V.x[:,1:end-1])/Δ.y + (V.y[2:end,:] - V.y[1:end-1,:])/Δ.x ) 
    @. τ.xx  = 2.0*ηs.c*ε̇.xx
    @. τ.yy  = 2.0*ηs.c*ε̇.yy
    @. τ.xy  = 2.0*ηs.v*ε̇.xy
    @. qD.x[2:end-1,:] = -k_ηf.x[2:end-1,:] * (Pf[2:end,:]-Pf[1:end-1,:])/Δ.x
    @. qD.y[:,2:end-1] = -k_ηf.y[:,2:end-1] * (Pf[:,2:end]-Pf[:,1:end-1])/Δ.y
    @. divqD = (qD.x[2:end,:] - qD.x[1:end-1,:])/Δ.x + (qD.y[:,2:end] - qD.y[:,1:end-1])/Δ.y  
    @. Fm.x[2:end-1,:] = - ((τ.xx[2:end-0,:]-τ.xx[1:end-1,:])/Δ.x + (τ.xy[2:end-1,2:end]-τ.xy[2:end-1,1:end-1])/Δ.y - (Pt[2:end,:]-Pt[1:end-1,:])/Δ.x)
    @. Fm.y[:,2:end-1] = - ((τ.yy[:,2:end-0]-τ.yy[:,1:end-1])/Δ.y + (τ.xy[2:end,2:end-1]-τ.xy[1:end-1,2:end-1])/Δ.x - (Pt[:,2:end]-Pt[:,1:end-1])/Δ.y)
    @. FPt             = divVs + (Pt-Pf)/ηb 
    @. FPf             = divqD - (Pt-Pf)/ηb 
    @printf("Residuals:\n")
    @printf("Fmx = %1.4e\n", norm(Fm.x)/length(Fm.x))
    @printf("Fmy = %1.4e\n", norm(Fm.y)/length(Fm.y))
    @printf("Fpt = %1.4e\n", norm(FPt)/length(FPt))
    @printf("Fpf = %1.4e\n", norm(FPf)/length(FPf))
end

function ResidualsNonLinear!(Fm, FPt, FPf, V, Pt, Pf, divVs, divqD, ε̇, τ, qD, ηs, ηb, k_ηf, Δ, BC, VxBC, VyBC, ϕ, ϕold, dt  )
    @. V.x[:,1]   = (BC.S==:Neumann)*V.x[:,2]     + (BC.S==:Dirichlet)*(2*VxBC.S - V.x[:,2])
    @. V.x[:,end] = (BC.N==:Neumann)*V.x[:,end-1] + (BC.N==:Dirichlet)*(2*VxBC.N - V.x[:,end-1])
    @. V.y[1,:]   = (BC.W==:Neumann)*V.y[2,:]     + (BC.W==:Dirichlet)*(2*VyBC.W - V.y[2,:])
    @. V.y[end,:] = (BC.E==:Neumann)*V.y[end-1,:] + (BC.E==:Dirichlet)*(2*VyBC.E - V.y[end-1,:])
    @. divVs = (V.x[2:end,2:end-1] - V.x[1:end-1,2:end-1])/Δ.x + (V.y[2:end-1,2:end] - V.y[2:end-1,1:end-1])/Δ.y
    @. ϕ.c     = 1 - exp(log10(1-ϕold) - divVs * dt)
    @. ϕ.x    = (ϕ.c[2:end,:] + ϕ.c[1:end-1,:]) / 2.0
    @. ϕ.y    = (ϕ.c[:,2:end] + ϕ.c[:,1:end-1]) / 2.0
    @. ε̇.xx  = (V.x[2:end,2:end-1] - V.x[1:end-1,2:end-1])/Δ.x - 1.0/3.0*divVs
    @. ε̇.yy  = (V.y[2:end-1,2:end] - V.y[2:end-1,1:end-1])/Δ.y - 1.0/3.0*divVs
    @. ε̇.xy  = 0.5*( (V.x[:,2:end] - V.x[:,1:end-1])/Δ.y + (V.y[2:end,:] - V.y[1:end-1,:])/Δ.x ) 
    @. τ.xx  = 2.0*ηs.c*ε̇.xx
    @. τ.yy  = 2.0*ηs.c*ε̇.yy
    @. τ.xy  = 2.0*ηs.v*ε̇.xy
    @. qD.x[2:end-1,:] = -k_ηf.x[2:end-1,:] * ϕ.x^1 * (Pf[2:end,:]-Pf[1:end-1,:])/Δ.x
    @. qD.y[:,2:end-1] = -k_ηf.y[:,2:end-1] * ϕ.y^1 * (Pf[:,2:end]-Pf[:,1:end-1])/Δ.y
    @. divqD = (qD.x[2:end,:] - qD.x[1:end-1,:])/Δ.x + (qD.y[:,2:end] - qD.y[:,1:end-1])/Δ.y  
    @. Fm.x[2:end-1,:] = - ((τ.xx[2:end-0,:]-τ.xx[1:end-1,:])/Δ.x + (τ.xy[2:end-1,2:end]-τ.xy[2:end-1,1:end-1])/Δ.y - (Pt[2:end,:]-Pt[1:end-1,:])/Δ.x)
    @. Fm.y[:,2:end-1] = - ((τ.yy[:,2:end-0]-τ.yy[:,1:end-1])/Δ.y + (τ.xy[2:end,2:end-1]-τ.xy[1:end-1,2:end-1])/Δ.x - (Pt[:,2:end]-Pt[:,1:end-1])/Δ.y)
    @. FPt             = divVs + (Pt-Pf)/ηb / (1-ϕ.c) 
    @. FPf             = divqD - (Pt-Pf)/ηb / (1-ϕ.c)
    @printf("Residuals:\n")
    @printf("Fmx = %1.4e\n", norm(Fm.x)/length(Fm.x))
    @printf("Fmy = %1.4e\n", norm(Fm.y)/length(Fm.y))
    @printf("Fpt = %1.4e\n", norm(FPt)/length(FPt))
    @printf("Fpf = %1.4e\n", norm(FPf)/length(FPf))
end