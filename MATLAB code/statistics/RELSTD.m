function[ S,names,L ] = RELSTD( X,subj,MIN,MAX,str )
%SD*



ui=unique(subj);
S=zeros(length(ui),size(X,2));
for i =1:length(ui)    
    for j=1:size(X,2)
        idx=find(subj==ui(i) & isnan(X(:,j))==0);
        S(i,j)=relativeSD(X(idx,j),MIN,MAX);
        L(i,j)=length(idx);
    end
end

base_str='SD^*_';
for j = 1:size(X,2)
    names{j}=[base_str '{' str{j} '}'];
end
    

end

