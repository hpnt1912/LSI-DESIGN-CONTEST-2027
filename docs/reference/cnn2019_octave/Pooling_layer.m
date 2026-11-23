% Pooling_layer function
%
%
% Input  : 'a_c'  output of Convolutional_layer, 'filter'  filter for convolution
% Output : 'a_p'  input of Mid_layer  
%


function a = Pooling_layer(In,stride)

    [In_Line,In_Low,In_Channel,k] = size(In);
    
    Out_Line = (In_Line - stride) / stride + 1;     % Number of output lines
    Out_Low = (In_Low - stride) / stride + 1;       % Number of output lows
    Out_Channel = In_Channel;
    
    a = zeros(Out_Line,Out_Low,Out_Channel,k);
    
    % MAX Pooling
        for line = 1 : Out_Line
            for low = 1 : Out_Low
                for channel = 1 : Out_Channel
                        LINE = (line-1) * stride + 1;
                        LOW = (low-1) * stride + 1;
                        a(line,low,channel,:) = max(max(In(LINE:LINE+stride-1,LOW:LOW+stride-1,channel,:)));
                end
            end
        end
    
end