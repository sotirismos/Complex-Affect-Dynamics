function [ S,names,L ] = PERC( X,subj,str )
%STD Summary of this function goes here
%   Detailed explanation goes here



ui=unique(subj);
S=zeros(length(ui),1);

add=1;
for i =1:length(ui)
    add=1;
    for j =1:size(X,2)
        idx=find(subj==ui(i) & isnan(X(:,j))==0);
        ss=sort(X(idx,j));
        for p = 0:100
            S(i,add)=ss(1+round((length(idx)-1)*p/100));
            add=add+1;
        end
        L(i,1)=length(idx);
    end
end
  
add=1;
for j =1:2
    for p=0:100 
        names{add}=['PERC_' str{j} '_' num2str(p)]; 
        add=add+1;
    end
end
            
    
end

