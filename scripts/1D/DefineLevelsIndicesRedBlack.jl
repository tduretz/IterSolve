function MainIndices(nx, ilev)

    
    x = zeros(Int,nx)
    
    if ilev>=1
        i1=1; i2=nx
        nxC = length(i1:2^(ilev-1):i2)
        for iRB=1:2
            if iseven(nxC) 
                seq = (i1+ilev*(iRB-1)):(ilev*(2)):(i2+ilev*(iRB-2))
            else 
                seq = (i1+ilev*(iRB-1)):(ilev*(2)):(i2-ilev*(iRB-1))
            end
            @show seq
            x[seq] .= iRB
        end
    end

    if ilev>1
        
        i1=2; i2=nx-1
        nxC = length(i1:2^(ilev-1):i2)
        for iRB=1:2
            if iseven(nxC) 
                seq = (i1+ilev*(iRB-1)):(ilev*(2)):(i2+ilev*(iRB-2))
            else 
                seq = (i1+ilev*(iRB-1)):(ilev*(2)):(i2-ilev*(iRB-1))
            end
            @show seq
            x[seq] .= iRB+2
        end
    end
    @show x
end

MainIndices(101, 1);