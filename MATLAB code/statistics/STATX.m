function [ S,names,L ] = STATX( X,subj )
%STD Summary of this function goes here
%   Detailed explanation goes here



ui=unique(subj);
S=zeros(length(ui),1);
for i =1:length(ui)    
    for j =1:size(X,2)
        idx=find(subj==ui(i) & isnan(X(:,j))==0);
        ss=sort(X(idx,j));
        if j ==1
            S(i,j)=ss(round(length(idx)*0.25));
        else
            S(i,j)=ss(round(length(idx)*0.75));
        end
        L(i,1)=length(idx);
    end
end
        
    
        
        
%     C=cov(X(idx,:));
%     C(find(isnan(C(:))==1))=0;
%     [V,D]=eig(C);
%     DD=diag(D);
%     S(i,1)=sum(abs(DD));
%     S(i,1)=sum(abs(C(:)));
%     S(i,1)=sum(sum(abs(C))); 

%     [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca(X(idx,:));
%     S(i,1)=sum(EXPLAINED(1:2));
    
names{1}='STATX_1';
names{2}='STATX_2';
    
end

