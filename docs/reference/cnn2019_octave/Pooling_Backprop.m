%  Pooling_Backprop Function
%  
%  Synopsys: [ ] = Mid_layer( , , , )
%  Input  : 'dz_m' Output of Mid_Backprop 
%  Output : 'dz_p'   Δz  

function  dz_p = Pooling_Backprop(a_c,stride,a_p,dz_o)

    [line,low,channel,k] = size(a_p);
    da2 = reshape(dz_o,[low,line,channel,k]);
    
    a_p = repelem(a_p,stride,stride);       % a_p を4×4に変換
    da2 = repelem(da2,stride,stride);       % da2 を4×4に変換
    
    [LINE,LOW,CHANNEL,k] = size(a_c);
    dz_p = zeros(LINE,LOW,CHANNEL,k);        % dz_p を a_c と同じサイズで生成
    
    dz_p(a_c == a_p) = 1;                        % a_p の要素は、a_c における最大プーリングの結果
    dz_p = dz_p .* da2;                          % a_c 内の最大値の部分だけが値を持つ da1 を生成
    
end