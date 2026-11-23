
% 判別したい入力を指定(1→黒、0→白)
IN(:,:,:,1) = [1 1 1;1 0 1;1 1 1];


    [A_C,Z_C] = Convolutional_layer(IN,filter_c,stride_c,bias_c,padding);   %畳み込み層
    A_P = Pooling_layer(A_C,stride_p);                                      %プーリング層
    [A_O,Z_O] = Output_layer(A_P,w_o,bias_o);                               %出力層
    
    disp(A_O);
    
    
    if(A_O(1,1) > A_O(2,1))
        fprintf('IN is 〇\n');
    end
    
    if(A_O(1,1) < A_O(2,1))
        fprintf('IN is ×\n');
    end
   
    
    