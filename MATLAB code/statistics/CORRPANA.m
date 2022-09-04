function [ S,names,L,nPositiveRho ] = CORRPANA( X,subj )
%CORRPANA



ui=unique(subj);
S=zeros(length(ui),1);
nPositiveRho=0;
for i =1:length(ui)    
    idx=find(subj==ui(i) & isnan(X(:,1))==0 & isnan(X(:,2))==0);
    S(i,1)=corr(X(idx,1),X(idx,2)); 
    if S(i,1)>0
        nPositiveRho=nPositiveRho+1;
%         S(i,1)=NaN;
    end
    L(i,1)=length(idx);
end

names{1}='\rho_{PANA}';
    
end

