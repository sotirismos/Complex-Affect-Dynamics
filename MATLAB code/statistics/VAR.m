function [ S,names ] = VAR( X,subj,str )
%VAR 

ui=unique(subj);
S=zeros(length(ui),size(X,2));
for i =1:length(ui)    
    for j=1:size(X,2)
        idx=find(subj==ui(i) & isnan(X(:,j))==0);
        S(i,j)=var(X(idx,j));
    end
end

base_str='VAR_';
for j = 1:size(X,2)
    names{j}=[base_str '{' str{j} '}'];
end
    


end

