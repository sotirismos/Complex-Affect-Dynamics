%% CORRELATIONS
close all
clear all
clc

load('results/initAll.mat')


%%%%%%%%%%%%%%%%%%%%%% COMPUTE CORRELATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% put all statistics in one matrix
SLong=[];
idUI=find(triu(ones(size(SAll{1},2)),1)==1);
for i =1:length(SAll)
    SLong=[SLong;zscore(SAll{i})];
    Ctemp=corr(SAll{i}); % correlations of individual data set
    CVector(:,i)=Ctemp(idUI);   
end

[C pval]=corr(SLong); % correlations of all data sets together

% test how similar correlation matrices are over data sets
for i = 1:size(CVector,2)
    goodness(1,i)=corr(CVector(:,i),C(idUI));
end
MEAN_MIN_MAX_GOODNESS=[mean(goodness') min(goodness') max(goodness')]
CRONBACH_ALPHA_CORRELATIONS=ICCEva(CVector, 'C-k') % cronbachs alpha


%%%%%%%%%%%%%%%%% HIERARCHICAL CLUSTERING %%%%%%%%%%%%%%%%%%%%%%%%%

% compute hierarchical clustering mainly for ordering in image
D=1-abs(C);
D=D-diag(diag(D));
Y=squareform(D,'tovector');
Z=linkage(Y,'complete');
[~,T,OP]=dendrogram(Z,'labels',names);
for i =1:length(names)
    names2{i}=names{OP(i)};
end
C2=C(OP,OP);
P2=pval(OP,OP);


%%%%%%%%%%% PLOT CORRELATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alphaFigure=1;
figure('position',[0 0 800*alphaFigure 600*alphaFigure])

filler=ones(length(C),length(C));
colorM=ones(length(C),length(C),3);
colorBase=zeros(length(C),length(C),3);
colorMin=colorBase;
colorMax=colorBase;

colorMax(:,:,1)=filler*0;
colorMax(:,:,2)=filler*0;
colorMax(:,:,3)=filler*0.5;

colorMin(:,:,1)=filler*0.5;
colorMin(:,:,2)=filler*0;
colorMin(:,:,3)=filler*0;

CCol=zeros(length(C),length(C),3);

posNumbers=C2.*heaviside(C2);
negNumbers=(-C2.*heaviside(-C2));
for i =1:3
    CCol(:,:,i)=CCol(:,:,i)+(posNumbers.*colorMax(:,:,i)+(1-posNumbers).*colorM(:,:,i)).*heaviside(C2);
    CCol(:,:,i)=CCol(:,:,i)+(negNumbers.*colorMin(:,:,i)+(1-negNumbers).*colorM(:,:,i)).*heaviside(-C2);
end

add=1;
for cr=-1:0.01:1
    if cr>0
        for i =1:3
            CB(add,i)=cr*colorMax(1,1,i)+(1-cr)*colorM(1,1,i);
        end
    else
        for i =1:3
            CB(add,i)=abs(cr)*colorMin(1,1,i)+(1-abs(cr))*colorM(1,1,i);
        end
    end
    add=add+1;
end

image(CCol)
axis square
set(gca,'xtick',[1:size(SAll{1},2)],'xticklabel',names2)
set(gca,'ytick',[1:size(SAll{1},2)],'yticklabel',names2)
xtickangle(90)
colormap(CB);
cbh=colorbar;
nTics=11;
cbh.Ticks = linspace(0, 1, nTics) ;
cbh.TickLabels = num2cell(linspace(-1, 1, nTics)) ;

saveas(gcf,'plots/CorrelationMatrix.emf')
print(gcf,'plots/CorrelationMatrix.pdf','-dpdf','-r0');

%%%%%%%%%%%%%%% SAVE IN TABLE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CTable=array2table(C);
CTable.Properties.VariableNames=namesX;
CTable.Properties.RowNames=namesX;
filename = 'results/CorrelationMatrix.xlsx';
writetable(CTable,filename,'Sheet',1,'Range','A2','WriteRowNames',true)