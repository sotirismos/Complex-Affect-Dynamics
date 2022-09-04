function [ S,names,L ] = STD( X,subj,str )
%STD 



ui=unique(subj);
S=zeros(length(ui),size(X,2));
for i =1:length(ui)    
    for j=1:size(X,2)
        idx=find(subj==ui(i) & isnan(X(:,j))==0);
        L(i,j)=length(idx);
        S(i,j)=std(X(idx,j));
    end
end

base_str='SD_';
for j = 1:size(X,2)
    names{j}=[base_str '{' str{j} '}'];
end
    



end

