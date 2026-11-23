% OX Judgement Program (Convolutional Neural Network) 
% LSI Design Contest in Okinawa 2020
%
% tsukaikata
%
% see double
%
clear all
MARU = [1 1 1;1 0 1;1 1 1];   % 〇
BATU = [1 0 1;0 1 0;1 0 1];   % ×

filter_c(:,:,1) = [1 1;1 -1];    % filter１
filter_c(:,:,2) = [1 -1;-1 1];   % filter２

%　Teacher data of 〇　（〇の教師データ）
In(:,:,:,1) = [1 1 1;1 0 1;1 1 1];
In(:,:,:,2) = [1 1 1;1 0 0;1 1 1];
In(:,:,:,3) = [1 1 1;1 0 1;1 1 0];
for i = 1 : 3
  t(:,i) = [1 0].';   % Correct answer when input is 〇（入力が〇の時の正解）
end

% Teacher data of ×（×の教師データ）
In(:,:,:,4) = [1 0 1;0 1 0;1 0 1];
In(:,:,:,5) = [1 0 0;0 1 0;1 0 1];
In(:,:,:,6) = [1 0 1;0 1 0;1 0 0];
for i = 4 : 6
  t(:,i) = [0 1].';   % Correct answer when input is ×　（入力が×の時の正解）
end

% Convolutional Layer
stride_c = 1;                                   % stride 
bias_c(:,:,1) = 1;            % bias
bias_c(:,:,2) = 1;
padding = 0;                                    % padding

% Poolong Layer
stride_p = 2;                       % stride

% Output Layer
w_o = [1.5 -2.6;-4.2 3.5];    % weight
bias_o = [1 1].';                   % bias

eta = 0.1;                          % 学習効率(η, eta)

for j=1:5000

    % Foward Propagation (フォワード処理)
    [a_c,z_c] = Convolutional_layer(In,filter_c,stride_c,bias_c,padding);                % Convolutional Layer (畳み込み層)
    a_p = Pooling_layer(a_c,stride_p);                                                   % Pooling Layer (プーリング層)
    [a_o,z_o] = Output_layer(a_p,w_o,bias_o);                                            % Output Layer (出力層)
    
    % Back Propagation (バックワード処理)
    [dz_o,dw_o,db_o,Ct] = Output_Backprop(a_o,t,a_p,w_o);                                % Output Layer (出力層)
    CT(j) = Ct;
    dz_p = Pooling_Backprop(a_c,stride_p,a_p,dz_o);                                      % Pooling Layer (プーリング層)
    [da_c,dw_c,db_c] = Convolutional_Backprop(In,filter_c,stride_c,padding,dz_p,a_c);    % Convolutional Layer (畳み込み層)

    % Update Value of Output Layer (出力層の重み、バイアスの値を更新)
    w_o = w_o - (sum(dw_o,3)*eta);             
    bias_o = bias_o - sum(db_o,3)*eta;                 

    % Update Value of Convolutional Layer (畳み込み層の重み、バイアスの値を更新)
    bias_c = bias_c - (sum(db_c,4)*eta);          
    filter_c = filter_c - (dw_c*eta);     

end
plot(CT);
xlabel('j');
ylabel('CT');
