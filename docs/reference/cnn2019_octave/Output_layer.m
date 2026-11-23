% Output_layer function
%
%
% Input  : 'a_m' output of Mid_Layer
% Output : 'a_o' output of Foward propagation

function [a,z_o] = Output_layer(a_p,w_o,bias_o)
    
     [line,low,channel,k] = size(a_p);
     
      z_o = zeros(channel,1,k);
     
     for  s = 1 : k
         z_o(1,1,s) = a_p(:,:,1,s)*w_o(1,1) + a_p(:,:,2,s)*w_o(2,1);
         z_o(2,1,s) = a_p(:,:,1,s)*w_o(1,2) + a_p(:,:,2,s)*w_o(2,2);
     end
     
     z_o = z_o + bias_o;
    
     a = 1./(1+exp(-z_o));
    
end