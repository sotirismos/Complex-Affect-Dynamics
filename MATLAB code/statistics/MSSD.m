function [  S,names,L ] = MSSD( X,subj,unit,beepno,str )
%MSSD 

idx1=1:(size(X,1)-1);
idx2=2:size(X,1);

ui=unique(subj);
S=zeros(length(ui),size(X,2));
for i =1:length(ui)    
    for j=1:size(X,2)
        idx=find(subj(idx1)==ui(i) & subj(idx2)==ui(i) &...
                 isnan(X(idx1,j))==0 & isnan(X(idx2,j))==0 &...
                 unit(idx1)==unit(idx2) & beepno(idx2)==(beepno(idx1)+1));                     
        L(i,j)=length(idx);
        S(i,j)=mean((X(idx2(idx),j)-X(idx1(idx),j)).^2);
%         S(i,j)=sum((X(idx2(idx),j)-X(idx1(idx),j)).^2)/(length(idx)-1);
    end
end

base_str='MSSD_';
for j = 1:size(X,2)
    names{j}=[base_str '{' str{j} '}'];
end
    



end

