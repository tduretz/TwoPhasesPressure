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
end

function ResidualsNonLinear!(Fm, FPt, FPf, V, Pt, Pf, divVs, divqD, ε̇, τ, qD, ηs, ηb, ηϕ, k_ηf, Δ, BC, VxBC, VyBC, ϕ, ϕ0, ϕold, k_ηf0, nϕ, dt, ηs_ini, ηb_ini, βs, ρs0, ρs, ρsold, P0  )
    @. V.x[:,1]   = (BC.S==:Neumann)*V.x[:,2]     + (BC.S==:Dirichlet)*(2*VxBC.S - V.x[:,2])
    @. V.x[:,end] = (BC.N==:Neumann)*V.x[:,end-1] + (BC.N==:Dirichlet)*(2*VxBC.N - V.x[:,end-1])
    @. V.y[1,:]   = (BC.W==:Neumann)*V.y[2,:]     + (BC.W==:Dirichlet)*(2*VyBC.W - V.y[2,:])
    @. V.y[end,:] = (BC.E==:Neumann)*V.y[end-1,:] + (BC.E==:Dirichlet)*(2*VyBC.E - V.y[end-1,:])
    @. divVs = (V.x[2:end,2:end-1] - V.x[1:end-1,2:end-1])/Δ.x + (V.y[2:end-1,2:end] - V.y[2:end-1,1:end-1])/Δ.y
    @. ρs    = ρs0 * exp(βs * (Pf - P0))
    @. ϕ.c   = 1.0 - exp(log(1.0 - ϕold)  - divVs * dt - log(ρs) + log(ρsold))
    # @. ϕ.c   = 1.0 - exp(log(1.0 - ϕold)  - divVs * dt)
    @. ϕ.x   = (ϕ.c[2:end,:] + ϕ.c[1:end-1,:]) / 2.0
    @. ϕ.y   = (ϕ.c[:,2:end] + ϕ.c[:,1:end-1]) / 2.0
    @. ε̇.xx  = (V.x[2:end,2:end-1] - V.x[1:end-1,2:end-1])/Δ.x - 1.0/3.0*divVs
    @. ε̇.yy  = (V.y[2:end-1,2:end] - V.y[2:end-1,1:end-1])/Δ.y - 1.0/3.0*divVs
    @. ε̇.xy  = 0.5*( (V.x[:,2:end] - V.x[:,1:end-1])/Δ.y + (V.y[2:end,:] - V.y[1:end-1,:])/Δ.x ) 
    ϕex = [hcat(ϕ.c[1,1], ϕ.c[1,:]', ϕ.c[1, end]); hcat(ϕ.c[:  ,1], ϕ.c',  ϕ.c[:  ,end]); hcat(ϕ.c[end,1], ϕ.c[end,:]', ϕ.c[end,end])]
    ϕv       = 0.25*(ϕex[1:end-1,1:end-1] + ϕex[2:end-0,1:end-1] + ϕex[1:end-1,2:end-0] + ϕex[2:end-0,2:end-0])
    # @. ηs.c  = ηs_ini.c * exp(-1 * (ϕ.c / ϕ0 -1))
    # @. ηs.v  = ηs_ini.v * exp(-1 * (ϕv / ϕ0 -1))
    @. ηs.c  = ηs_ini.c * (ϕ0 / ϕ.c)^3
    @. ηs.v  = ηs_ini.v * (ϕ0 / ϕv)^3
    @. τ.xx  = 2.0*ηs.c*ε̇.xx
    @. τ.yy  = 2.0*ηs.c*ε̇.yy
    @. τ.xy  = 2.0*ηs.v*ε̇.xy
    @. k_ηf.x[2:end-1,:] = k_ηf0 * ϕ.x^nϕ    # ACHTUNG: Store non linear coefficient (to be reused in matrix assembly for Picard linearisation)
    @. k_ηf.y[:,2:end-1] = k_ηf0 * ϕ.y^nϕ
    @. qD.x[2:end-1,:]   = - k_ηf.x[2:end-1,:] * (Pf[2:end,:]-Pf[1:end-1,:])/Δ.x
    @. qD.y[:,2:end-1]   = - k_ηf.y[:,2:end-1] * (Pf[:,2:end]-Pf[:,1:end-1])/Δ.y
    @. divqD = (qD.x[2:end,:] - qD.x[1:end-1,:])/Δ.x + (qD.y[:,2:end] - qD.y[:,1:end-1])/Δ.y  
    @. Fm.x[2:end-1,:] = - ((τ.xx[2:end-0,:]-τ.xx[1:end-1,:])/Δ.x + (τ.xy[2:end-1,2:end]-τ.xy[2:end-1,1:end-1])/Δ.y - (Pt[2:end,:]-Pt[1:end-1,:])/Δ.x)
    @. Fm.y[:,2:end-1] = - ((τ.yy[:,2:end-0]-τ.yy[:,1:end-1])/Δ.y + (τ.xy[2:end,2:end-1]-τ.xy[1:end-1,2:end-1])/Δ.x - (Pt[:,2:end]-Pt[:,1:end-1])/Δ.y)
    @. ηb    = ηb_ini * min(ϕ0 / ϕ.c, 100)
    @. ηϕ    = (1.0 - ϕ.c) * ηb     # ACHTUNG: Store non linear coefficient (to be rsused in matrix assembly for Picard linearisation)
    @. FPt   = divVs + (Pt-Pf)/ηϕ 
    @. FPf   = divqD - (Pt-Pf)/ηϕ
    
end