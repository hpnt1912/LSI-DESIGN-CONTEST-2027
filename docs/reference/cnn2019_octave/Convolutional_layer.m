% Convolutional_layer function
%
%
% Input  : 'In'  input data, 'filter'  filter for convolution
% Output : 'a_c' , 'z_c'  output of convolutional layer

function [a,z] = Convolutional_layer(In,filter,stride,bias,padding) 

    In = padarray(In,[padding padding],0,'both');   % zero padding
    
    In_Line = size(In,1);   % Number of input lines
    In_Low = size(In,2);    % Number of input lows
    
    filter_Line = size(filter,1);   % Number of filter lines
    filter_Low = size(filter,2);    % Number of filter lows

    Out_Line = (In_Line - filter_Line)/stride + 1;  % Number of output lines
    Out_Low = (In_Low - filter_Low)/stride + 1;     % Number of output lows
    Out_Channel = size(filter,3);                   % Number of output channels
    
    z = zeros(Out_Line,Out_Low,Out_Channel,size(In,4));
    for k=1 : size(In,4)
        for channel = 1 : Out_Channel   
            for low = 1 : Out_Low   
                for line = 1 :  Out_Line
                     LINE = (line-1) * stride + 1; 
                     LOW = (low-1) * stride + 1 ;
                     Out = In(LINE:LINE + filter_Line-1,LOW:LOW+filter_Low-1,:,k) .* filter(:,:,channel);
                     z(line,low,channel,k) = sum(sum(Out));
                end
            end
        end
    end
    
    z = z + bias;
    a = 1 ./ (1+exp(-z));
  
end