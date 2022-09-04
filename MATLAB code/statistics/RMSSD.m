function [  S,names,L ] = RMSSD( X,subj,unit,beepno,str )
%RMSSD

[S,~,L]=MSSD( X,subj,unit,beepno,str );
S=sqrt(S);

base_str='MSSD_';
for j = 1:size(X,2)
    names{j}=[base_str '{' str{j} '}'];
end
    



end

