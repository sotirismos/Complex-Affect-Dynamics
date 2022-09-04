function  [  S,names,L ] = DENSITY( X,subj,unit,beepno,dataStr )
%DENSITY

idx1=1:(size(X,1)-1);
idx2=2:size(X,1);

ui=unique(subj);
S=zeros(length(ui),1);


ui=unique(subj);
for i =1:length(ui)    
    for j=1:size(X,2)
        idxA=find(subj==ui(i) & isnan(X(:,j))==0);
        temp=mean(X(idxA,j));
        X(idxA,j)=X(idxA,j)-temp;
        
        idx=find(subj(idx1)==ui(i) & subj(idx2)==ui(i) &...
            unit(idx1)==unit(idx2) & beepno(idx2)==(beepno(idx1)+1));               
        Xtemp=[X(idx2(idx),j) X(idx1(idx),:)];
        idxUse=find(sum(isnan(Xtemp),2)==0);
        L(i,j)=length(idxUse);        
    end
end

for j=1:size(X,2)
    if exist(['data/' dataStr '/VAR' num2str(j) '.mat'])
        load(['data/' dataStr '/VAR' num2str(j) '.mat'])
        paramsA{j}=params;
        if (size(params,2)-1)~=size(X,2)
            error('Loaded wrong mixed model, wrong number of emotions')
        end
        
        if size(paramsA{1},1)~=length(unique(subj))
            error('Loaded wrong mixed model, wrong number of subjects')
        end
        
        
    else
        idx=find(subj(idx1)==subj(idx2) & unit(idx1)==unit(idx2) & beepno(idx2)==(beepno(idx1)+1));      
        Xtemp=[X(idx2(idx),j) X(idx1(idx),:)];
        idxUse=find(sum(isnan(Xtemp),2)==0);

        Xtrain=[ones(length(idxUse),1) X(idx1(idx(idxUse)),:)];
        Ytrain=X(idx2(idx(idxUse)),j);
        SubjTrain=subj(idx1(idx(idxUse)));

        mdl=fitlmematrix(Xtrain,Ytrain,Xtrain,SubjTrain);
        params=repmat(mdl.fixedEffects',length(ui),1)+reshape(mdl.randomEffects,size(Xtrain,2),length(ui))';          
        paramsA{j}=params;
        if(~exist(['/' dataStr]))
            mkdir(['C:\Users\sotir\Desktop\data\' dataStr]);
        end
        save(['C:\Users\sotir\Desktop\data\' dataStr '\VAR' num2str(j) '.mat'],'mdl','params');
    end
end

A=zeros(size(X,2),size(X,2));
for i =1:length(ui)
    for j=1:size(X,2)
        A(j,:)=paramsA{j}(i,2:end);
    end
    S(i,1)=mean(abs(A(:)));
end

names{1}='D';


end

