function   [result]=FDSL_MC(X,gt,option)
% ,lambda,gamma
%%----------------Initialize-------------------
numClust=option.numClust; 
K=option.K;
threshold=option.threshold;
delta=option.delta;
lambda=option.lambda;         
r=option.r;
max_iter=option.max_iter;
Vnum=option.Vnum;
N=option.N;                             
alpha=option.alpha; 
beta=option.beta; 
gamma=option.gamma;
alpha_r=alpha.^r;
Jlast=99999;
IsConverge = 0;
iter = 1;
Q=cell(size(X,1),size(X,2));
U=cell(size(X,1),size(X,2));
H=cell(size(X,1),size(X,2)); 
Y=cell(size(X,1),size(X,2));
V1=zeros(K,K);
V2=zeros(K,N);
V=rand(K,N);
J1=ones(Vnum,1);
J2=ones(Vnum,1);
J4=ones(Vnum,1);
Ja=ones(Vnum,1);
tic;
for i=1:Vnum 
    Q{i}=rand(numClust,K);
    U{i}=rand(size(X{i},1),K);
    H{i}=rand(N,K); 
    Y{i}=rand(numClust,N);
    YY=litekmeans(X{i}',numClust,'MaxIter', 100);
    Y{i}=ToM(YY,size(Y{i},1),size(Y{i},2));
end 
%%-------------construct Laplaican
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 10;
    options.WeightMode = 'HeatKernel';      % Binary  HeatKernel Cosine

%D = squareform(D);
for i=1:Vnum
  W{i} = constructW(X{i}',options);
  D{i} = diag(sum(W{i},1));
  L{i} = D{i} - W{i};
end



C=rand(K,numClust);
g0=ceil(rand(1,N)*numClust);
% G1=ToM(g0',numClust,size(V,2));  
% G2=ToM(g0',numClust,size(V,2));  
G=ToM(g0',numClust,size(V,2));  

%%------------------Update---------------------------

%%
while (IsConverge == 0&&iter<max_iter+1) 
    
    %---------UpdateQi---------
     for i=1:Vnum
        Q{i}=Y{i}*V'/(V*V'+lambda/(alpha_r(i)*delta)*eye(size(V,1)));
     end
     %---------UpdateUi---------
     for i=1:Vnum
      U{i}=(X{i}*V'+X{i}*H{i})/(V*V'+H{i}'*H{i}+lambda/(alpha_r(i))*eye(size(V,1))); %新加
     end     
      %---------UpdateHi---------
     for i=1:Vnum  
         H{i}=H{i}.*((alpha_r(i)*X{i}'* U{i}+gamma*W{i}*H{i})./(alpha_r(i)*H{i}*U{i}'*U{i}+gamma*D{i}*H{i}+lambda*H{i}));
     end  %新加
    %---------UpdateV---------
    V1=zeros(K,K);
    V2=zeros(K,N);
     for i=1:Vnum
        V1=V1+(alpha_r(i)*U{i}'*U{i}+alpha_r(i)*delta*Q{i}'*Q{i});
        V2=V2+(alpha_r(i)*U{i}'*X{i}+alpha_r(i)*delta*Q{i}'*Y{i});
     end  
     V=(V1+(beta+lambda)*eye(K))\(V2+beta*C*G);
     
     C=V*G'/(G*G'+lambda/beta*eye(size(numClust)));
     %---------UpdateG---------
     for i = 1:N
        xVec = V(:,i);
        G(:,i) = findindicator(xVec, C);
     end
%%
    %---------Calculate J---------
    J3=norm(V-C*G,'fro')^2;
    for i=1:Vnum 
        J1(i)=alpha_r(i)*norm(X{i}-U{i}*V,'fro')^2;
        J2(i)=alpha_r(i)*norm(Y{i}-Q{i}*V,'fro')^2;
        J_new(i)=alpha_r(i)*norm(X{i}-U{i}*H{i}','fro')^2;
        j_new2(i)= trace(H{i}'*L{i}*H{i});
        J4(i)=norm(U{i},'fro')^2+norm(Q{i},'fro')^2+norm(H{i},'fro')^2+norm(L{i},'fro')^2;
        Ja(i)=J1(i)+delta*J2(i)+J_new(i); 
    end
    J5=(norm(V,'fro')^2+norm(C,'fro')^2)+sum(J4);
    Jcurrent=sum(J1)+delta*sum(J2)+sum(J_new)+beta*J3+lambda*J5+gamma*sum(j_new2);
    %---------Calculate alpha---------   
    alpha=(Ja.^(1/(1-r)))/sum(Ja.^(1/(1-r)));
    alpha_r=alpha.^r;
    
    %---------Iscoverage---------   
    J(iter)=Jcurrent;
    if (abs(Jlast - Jcurrent)) < threshold
            IsConverge=1;
    end
    Jlast=Jcurrent;
    iter=iter+1; 
end
%%
toc
[~,pred_label] = max(G,[],1);   
pred_label=pred_label';
 result = ClusteringMeasure(gt, pred_label);                           
 [f,p,r] = compute_f(gt,pred_label);                                   
 [A nmi avgent] = compute_nmi(gt,pred_label);
 result=[result f p r];
end


