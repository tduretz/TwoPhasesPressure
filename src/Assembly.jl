function Assembly( ηs, ηb, k_ηf, BC, Num, nv, nc, Δ )
    # Linear system of equation
    ndof   = maximum(Num.Pf)
    K      = ExtendableSparseMatrix(ndof, ndof)
    dx, dy = Δ.x, Δ.y

    #############################
    # Total momentum equation x #
    #############################
    for i=1:nv.x, j=1:nc.y
        if i==1 || i==nv.x
            # Equation number
            ii = Num.Vx[i,j]
            # Linear system coefficients
            K[ii,ii] = 1.0
        else
            # Equation number
            ii = Num.Vx[i,j]
            # Stencil
            iS = ii - nv.x 
            iW = ii - 1 
            iC = ii
            iE = ii + 1
            iN = ii + nv.x
            #------------------#
            iSW = Num.Vy[i-1,j]
            iSE = iSW + 1
            iNW = iSW + nc.y
            iNE = iSW + nc.y + 1
            #------------------#
            iPW = Num.Pt[i-1,j]
            iPE = Num.Pt[i,j]
            #------------------#
            DirS = NeuS = DirN = NeuN = 0
            DirS = (j==1 && BC.S==:Dirichlet) 
            NeuS = (j==1 && BC.S!=:Dirichlet) 
            DirN = (j==nc.x && BC.N==:Dirichlet) 
            NeuN = (j==nc.x && BC.N!=:Dirichlet)
            iS   = (j==1)    ? 1 : iS
            iN   = (j==nc.x) ? 1 : iN
            #------------------#
            eW   = ηs.c[i-1,j]
            eE   = ηs.c[i,j]
            eS   = ηs.v[i,j]
            eN   = ηs.v[i,j+1]
            #------------------#
            K[ii,iS]  = -eS .* (-DirS - NeuS + 1) ./ dy .^ 2
            K[ii,iW]  = -4 // 3 * eW ./ dx .^ 2
            K[ii,iC]  = -(2 * eN .* (-DirN ./ dy - (-DirN - NeuN + 1) ./ (2 * dy)) - 2 * eS .* (DirS ./ dy + (-DirS - NeuS + 1) ./ (2 * dy))) ./ dy - (-4 // 3 * eE ./ dx - 4 // 3 * eW ./ dx) ./ dx
            K[ii,iE]  = -4 // 3 * eE ./ dx .^ 2
            K[ii,iN]  = -eN .* (-DirN - NeuN + 1) ./ dy .^ 2
            K[ii,iSW] = -eS ./ (dx .* dy) + (2 // 3) * eW ./ (dx .* dy)
            K[ii,iSE] = -2 // 3 * eE ./ (dx .* dy) + eS ./ (dx .* dy)
            K[ii,iNW] = eN ./ (dx .* dy) - 2 // 3 * eW ./ (dx .* dy)
            K[ii,iNE] = (2 // 3) * eE ./ (dx .* dy) - eN ./ (dx .* dy)
            K[ii,iPW] = -1 ./ dx
            K[ii,iPE] = 1 ./ dx
        end
    end

    #############################
    # Total momentum equation y #
    #############################
    for i=1:nc.x, j=1:nv.y
        if j==1 || j==nv.y
            ii = Num.Vy[i,j]
            K[ii,ii] = 1.0
        else
            # Equation number
            ii = Num.Vy[i,j]
            # Stencil
            iS = ii - nc.x 
            iW = ii - 1 
            iC = ii
            iE = ii + 1
            iN = ii + nc.x
            #------------------#
            iSW = Num.Vx[i,j-1]
            iSE = iSW + 1
            iNW = iSW + nv.x     
            iNE = iSW + nv.x + 1
            #------------------#
            iPS = Num.Pt[i,j-1]
            iPN = Num.Pt[i,j]
            #------------------#            
            DirW = NeuW = DirE = NeuE = 0
            DirW = (i==1 && BC.W==:Dirichlet) 
            NeuW = (i==1 && BC.W!=:Dirichlet) 
            DirE = (i==nc.y && BC.E==:Dirichlet) 
            NeuE = (i==nc.y && BC.E!=:Dirichlet)
            iW   = (i==1)    ? 1 : iW
            iE   = (i==nc.y) ? 1 : iE       
            #------------------#
            eW   = ηs.v[i,j]
            eE   = ηs.v[i+1,j]
            eS   = ηs.c[i,j-1]
            eN   = ηs.c[i,j]            
            #------------------#
            K[ii,iS]  = -4 // 3 * eS ./ dy .^ 2
            K[ii,iW]  = -eW .* (-DirW - NeuW + 1) ./ dx .^ 2
            K[ii,iC]  = -(-4 // 3 * eN ./ dy - 4 // 3 * eS ./ dy) ./ dy - (2 * eE .* (-DirE ./ dx - (-DirE - NeuE + 1) ./ (2 * dx)) - 2 * eW .* (DirW ./ dx + (-DirW - NeuW + 1) ./ (2 * dx))) ./ dx
            K[ii,iE]  = -eE .* (-DirE - NeuE + 1) ./ dx .^ 2
            K[ii,iN]  = -4 // 3 * eN ./ dy .^ 2
            K[ii,iSW] = (2 // 3) * eS ./ (dx .* dy) - eW ./ (dx .* dy)
            K[ii,iSE] = eE ./ (dx .* dy) - 2 // 3 * eS ./ (dx .* dy)
            K[ii,iNW] = -2 // 3 * eN ./ (dx .* dy) + eW ./ (dx .* dy)
            K[ii,iNE] = -eE ./ (dx .* dy) + (2 // 3) * eN ./ (dx .* dy)
            K[ii,iPS] = -1 ./ dy
            K[ii,iPN] = 1 ./ dy     
        end    
    end

    #############################
    # Solid continuity equation #
    #############################
    for i=1:nc.x, j=1:nc.y
        # Equation number
        ii = Num.Pt[i,j]
        # Stencil
        iW = Num.Vx[i,j]
        iE = Num.Vx[i+1,j]
        iS = Num.Vy[i,j]
        iN = Num.Vy[i,j+1]
        # Material coefficient
        e_phi = ηb[i,j]
        # Linear system coefficients
        K[ii,ii] = 1 ./ e_phi
        K[ii,iW] = -1 ./ dx
        K[ii,iE] = 1 ./ dx
        K[ii,iS] = -1 ./ dy 
        K[ii,iN] = 1 ./ dy
        iPf = Num.Pf[i,j]
        K[ii,iPf] = -1 ./ e_phi
    end

    #############################
    # Fluid continuity equation #
    #############################
    for i=1:nc.x, j=1:nc.y
        # Equation number
        ii = Num.Pf[i,j]
        # Stencil
        iS = ii - nc.x
        iW = ii - 1
        iC = ii
        iE = ii + 1
        iN = ii + nc.x
        # Boundaries
        iW     = i==1    ? 1  : iW   
        BCPfW  = i==1    ? 1. : 0.
        iE     = i==nc.x ? 1  : iE   
        BCPfE  = i==nc.x ? 1. : 0.
        iS     = j==1    ? 1  : iS   
        BCPfS  = j==1    ? 1. : 0.
        iN     = j==nc.y ? 1  : iN   
        BCPfN  = j==nc.y ? 1. : 0.
        # Material coefficient
        k_ef_W = k_ηf.x[i,j]
        k_ef_E = k_ηf.x[i+1,j]
        k_ef_S = k_ηf.y[i,j]
        k_ef_N = k_ηf.y[i,j+1]
        e_phi  = ηb[i,j]
        # Linear system coefficients
        K[ii,iS] = -k_ef_S .* (1 - BCPfS) ./ dy .^ 2
        K[ii,iW] = -k_ef_W .* (1 - BCPfW) ./ dx .^ 2
        K[ii,iC] = (k_ef_N .* (1 - BCPfN) ./ dy + k_ef_S .* (1 - BCPfS) ./ dy) ./ dy + (k_ef_E .* (1 - BCPfE) ./ dx + k_ef_W .* (1 - BCPfW) ./ dx) ./ dx + 1. /e_phi
        K[ii,iE] = -k_ef_E .* (1 - BCPfE) ./ dx .^ 2
        K[ii,iN] = -k_ef_N .* (1 - BCPfN) ./ dy .^ 2
        iPf = Num.Pt[i,j]
        K[ii,iPf] = -1. /e_phi
    end
    return flush!(K)
end