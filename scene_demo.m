clear;
addpath('./dataset');
rand('seed',700);
load('scene.mat')
for i=1:size(X,2)
    X{i}=mapminmax(X{i},-1,1);
%     X{i}=NormalizeData(X{i});
    N = size(X{i},2);
end

option.numClust = size(unique(gt),1);

option.N = size(X{1},2); 
option.K=100;
option.threshold=1e-1;   

option.delta=8e-1; 
option.beta=9e-3;
option.lambda=1e-2;
option.gamma=1e-1;

option.r=5;
option.max_iter = 20; 
option.Vnum = size(X,2);                          
option.alpha = ones(option.Vnum,1) / (option.Vnum); 

[result]=PLCMF_Laplican(X,gt,option); 

fprintf('\nscene: ACC = %.4f, NMI = %.4f, Purity = %.4f, F-score = %.4f\n',result(1),result(2),result(3),result(4));

% scene: ACC = 0.7470, NMI = 0.6042, Purity = 0.7470, F-score = 0.5929, Precision = 0.5902 and , Recall = 0.5957


