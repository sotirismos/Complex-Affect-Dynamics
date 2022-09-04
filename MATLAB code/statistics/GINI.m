function [ S,names,L ] = GINI( X,subj,Val,str,limit)
%GINI



ui=unique(subj);
uiV=unique(Val);
S=zeros(length(ui),1);
for iv=1:length(uiV)
    idxV=find(Val==uiV(iv));
    Xt=X(:,idxV);
    for i =1:length(ui)    
        idx=find(subj==ui(i));
        for k=1:size(Xt,2)
            idxNan=find(isnan(Xt(idx,k))==1);
            idx=setdiff(idx,idx(idxNan));
    %         idx
        end   
        L(i,iv)=length(idx);
        S(i,iv)=giniEmodiversity(Xt(idx,:), limit);    
    end
end

base_str='G_';
add=1;
for j = size(uiV,2):-1:1
    names{add}=[base_str '{' str{j} '}'];
    add=add+1;
end
    
end

