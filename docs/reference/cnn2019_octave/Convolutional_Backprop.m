%  Convolutional_Backprop Function
%  
%  Synopsys: [ , , ] = Mid_layer( , , , , , )
%  Input  : 'In' input data , 'dz_p' Output of Pooling_Backprop 
%  Output : 'dCda' , 'dCdw' , 'dCdb'  Δa , Δw , Δbias 

function [da1,dw,db] = Convolutional_Backprop(In,filter,stride,padding,dz_p,a_c)

    [line,low,channel,K] = size(In);
    [f_line,f_low,f_channel] = size(filter);
    [z_line,z_low,z_channel,z_k] = size(dz_p);
    
    dz2 = dz_p .* a_c .* (1-a_c);
    
    %db = sum(dz2,4)/K;
     db = sum(sum(dz2,1),2);
    
    h = z_line + (stride - 1) * (z_line - 1);
    w = z_low + (stride - 1) * (z_low - 1);
    c = z_channel;
    d = zeros(h,w,c,K);
    
    d(1:stride:h,1:stride:w,:,:) = dz2;
    dz2 = reshape(dz2,[f_line,f_low,f_channel,K]);

    d = padarray(d,[f_line-padding-1 f_low-padding-1],0,'both');

    da1 = zeros(line,low,f_channel,K);

    df = rot90(filter,2);
    
    
    for k = 1 : f_channel
         for j = 1 : low
             for i = 1 : line
                 a = d(i:i+f_line-1,j:j+f_low-1,:,:) .* df(:,:,k);
                 da1(i,j,k,:) = sum(sum(sum(a,1),2),3);
             end
         end
    end
    
    dw = zeros(f_line,f_low,f_channel);

    In = padarray(In,[padding padding],0,'both');
    
    for k = 1 : f_channel
       for j = 1 : f_low
           for i =1 : f_line
               h_start = (i-1) * stride + 1;
               w_start = (j-1) * stride + 1;
               dw(i,j,k,:) = sum(sum(sum(dz2(:,:,k,:) .* In(h_start:h_start+z_line-1,w_start:w_start+z_low-1,:,:),1),2),4);
           end
       end
    end
    

end
