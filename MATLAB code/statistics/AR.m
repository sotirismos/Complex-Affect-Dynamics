function [  S,names,L ] = AR(  X,subj,unit,beepno,str )
%AR

idx1=1:(size(X,1)-1);
idx2=2:size(X,1);

ui=unique(subj);
S=zeros(length(ui),size(X,2));


ui=unique(subj);
S=zeros(length(ui),size(X,2));
for i =1:length(ui)    
    for j=1:size(X,2)
        idx=find(subj==ui(i) & isnan(X(:,j))==0);
        
% old
        temp=mean(X(idx,j));
        X(idx,j)=X(idx,j)-temp;
        
        idxUse=find(isnan(X(idx1,j))==0 & isnan(X(idx2,j))==0 &...
             subj(idx1)==ui(i) & subj(idx2)==ui(i) & ...
             unit(idx1)==unit(idx2) & beepno(idx2)==(beepno(idx1)+1));
         
%          idxX=unique([idx1(idxUse) idx2(idxUse)])
%         temp=mean(X(idxX,j));
%         X(idxX,j)=X(idxX,j)-temp;
         
         
         L(i,j)=length(idxUse);
        
    end
end


% nanmean(X)
% pause;
for j=1:size(X,2)
    idx=find(isnan(X(idx1,j))==0 & isnan(X(idx2,j))==0 &...
             subj(idx1)==subj(idx2) & ...
             unit(idx1)==unit(idx2) & beepno(idx2)==(beepno(idx1)+1));      
         
    Xtrain=[ones(length(idx),1) X(idx1(idx),j)];
    Ytrain=X(idx2(idx),j);
    
    mdl1=fitlmematrix(Xtrain,Ytrain,Xtrain,subj(idx1(idx)));
    
    params=repmat(mdl1.fixedEffects',length(ui),1)+reshape(mdl1.randomEffects,size(Xtrain,2),length(ui))';            
    S(:,j)=params(:,2);
end


base_str='AR_';
for j = 1:size(X,2)
    names{j}=[base_str '{' str{j} '}'];
end
    





end

