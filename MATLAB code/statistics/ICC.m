function [ S,names,L,nNegative ] = ICC( X,subj,Val,str )
%ICC



ui=unique(subj);
uiV=unique(Val);
S=zeros(length(ui),1);
nNegative=0;
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
        
        iccTemp=ICCEva(Xt(idx,:), 'C-k');
        if iccTemp<0
             S(i,iv)=0;
             nNegative=nNegative+1;
        else
            S(i,iv)=iccTemp;
        end
        
%         S(i,iv)=max(0,ICCEva(Xt(idx,:), 'C-k'));  % put negative on zero    
%          temp=ICCEva(Xt(idx,:), 'C-k');
%          if temp<0
%              S(i,iv)=NaN;
%          else
%             S(i,iv)=temp;
%          end
    end
end

base_str='ICC_';
add=1;
for j = size(uiV,2):-1:1
    names{add}=[base_str '{' str{j} '}'];
    add=add+1;
end
    
    
end

