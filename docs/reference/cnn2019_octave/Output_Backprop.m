%  Output_Backprop Function
%  
%  Synopsys: [ , , , ] = Mid_layer( , , , )
%  Input  : 'a_o' Output of Output_layer 
%  Output : 'dCda' , 'dCdw' , 'dCdb'  Δa , Δw , Δbias 

function [dCda,dCdw,dCdb,Ct] = Output_Backprop(a_o,t,a_p,w_o)

    [line,low,channel,k] = size(a_p);
    
    for i = 1 : k
        C(i,1) = sum((t(:,i) - a_o(:,:,i)).^2)/2;
    end
    
    Ct = sum(C);
    for i = 1 : k
        dCdo(:,:,i) = a_o(:,:,i) - t(:,i);
        dCdz(:,:,i) = dCdo(:,:,i) .* a_o(:,:,i) .* (1-a_o(:,:,i));
    end

    dCdb = dCdz;

    for s = 1 : k
        for i = 1 : channel
           dCdw(i,1,s) = dCdz(1,:,s)*a_p(:,:,i,s);
           dCdw(i,2,s) = dCdz(2,:,s)*a_p(:,:,i,s);
        end
    end

    for s = 1 : k
        for i = 1 : channel
            dCda(i,1,s) = (dCdz(1,:,s)*w_o(i,1) + dCdz(2,:,s)*w_o(i,2)) * a_p(:,:,i,s) * (1-a_p(:,:,i,s));
        end
    end

end