%% CREATE PCA
close all
clear all
clc

%%%%%%%%% DECLARATION PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nComponents=6; % keep n components
fontsize=9; % fontsize image
cutOffPCA=0.2; % only show pca if larger than

%HARD CODED PARTS
differentSign=[5 4 6 3]; % these components get different signs
order=[4 5 3 1 2 6];  % To reorder components
nameCompsOrdered={'M_P','SD_P',['M_N and SD_N'],['SD*_N'],...
    'Time','\rho_{PN}'} % Names of components
ppp=[0.99 1 0.995 0.99 0.999 0.997] % choose percentiles with nice figure

% LOAD DATA
load('results/initAll.mat')
load('results/allSeries.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%% COMPUTE PCA %%%%%%%%%%%%%%%%%%%%%%%%%%%
% put all statistics in one matrix
SLong=[];
for i =1:length(SAll)
    SLong=[SLong;zscore(SAll{i})];
end
SLongZ=zscore(SLong);

% PCA number factors using parallel analysis
eigPar=ParallelAnalysis(SLongZ);
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca(SLongZ);
plot(eigPar,'.-')
hold on
plot(LATENT, '.-');
legend('95% percentile random data','observed');
xlabel('Component number');
ylabel('Eigenvalue')
saveas(gcf,'plots/ploteScree.pdf')
saveas(gcf,'plots/ploteScree.emf')
% using this parallel analysis only 4 components should be chosen. As with
% in the main text we show the result with 6 components

% compute and rotate PCA
% https://stats.stackexchange.com/questions/59213/how-to-compute-varimax-rotated-principal-components-in-r
LOADINGS=COEFF*diag(sqrt(LATENT));
[LOADINGROT, T] = rotatefactors(LOADINGS(:,1:nComponents),'Method','varimax');
COEFFB=COEFF(:,1:nComponents)*diag(1./sqrt(LATENT(1:nComponents)))*T;

% give good direction
for i = 1:length(differentSign)
    COEFFB(:,differentSign(i))=-COEFFB(:,differentSign(i));
    LOADINGROT(:,differentSign(i))=-LOADINGROT(:,differentSign(i));
end

% scores pca
SPCA=SLongZ*COEFFB;

% ORDER PCA
LOADINGROTOrder=LOADINGROT(:,order);
SPCA=SPCA(:,order);

%%%%%%%%%%%%%%%%%%%% PLOT PCA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha=0.8;
figure('position',[0 0 1000*alpha 1100*alpha])
co=get(gca,'colororder');

letters={'A.','B.'};

% loop over components
for j =1:nComponents

subplot(nComponents,2,(j-1)*2+1);
stem(LOADINGROTOrder(:,j),'filled','markersize',2)
rotVar=60;

% loop over coefficients
for i =1:size(SLongZ,2)
    if (abs(LOADINGROTOrder(i,j))>cutOffPCA)
        t=text(i+0.1,LOADINGROTOrder(i,j),names{i},'FontSize',7)

        if(LOADINGROTOrder(i,j))>0
            set(t,'rotation',rotVar)
        else
            set(t,'rotation',-rotVar)
        end
        set(t,'fontsize',6)
    end
end

% layout plot
x0=0;
xe=size(SLongZ,2)+1;
y0=-1.6;
ye=2.1;
axis([x0 xe y0 ye])
if j==nComponents
    text(x0-(xe-x0)/5*0.8,y0-(ye-y0)/2.5,letters{1},'FontSize',30);
end

text(x0-(xe-x0)/6*0.8,y0-(ye-y0)/100,[num2str(j) '.'],'FontSize',20);



xticks([]);
ylabel('Loading')
if j ==nComponents
    xlabel('affect dynamic measures');
end
if j==1
    title('Component loadings');
end
set(gca,'ytick',[-1 1])
ylabel([nameCompsOrdered{j} newline 'loading'])
set(gca,'fontsize', fontsize)


% plot example with high score
toSort=SPCA(:,j)+0.1*mean(abs(SPCA(:,[1:j-1 j+1:size(SPCA,2)])),2);
[~,ids]=sort(SPCA(:,j));
idHigh=ids(round(length(ids)*ppp(j)));
subplot(nComponents,2,(j-1)*2+2);
idnonan=find(isnan(sum(personTimeSeries{idHigh,1},2))==0);
plot(mean(personTimeSeries{idHigh,2}(idnonan,:),2),'color',co(1,:))
hold on
idnonan=find(isnan(sum(personTimeSeries{idHigh,1},2))==0);
plot(mean(personTimeSeries{idHigh,1}(idnonan,:),2),'color',co(2,:))

% layout
x0=1;
xe=length(idnonan);
y0=-5;
ye=105;
axis([x0 xe y0 ye])
if j==nComponents
    text(x0-(xe-x0)/5*0.8,y0-(ye-y0)/2.5,letters{2},'FontSize',30);
end
ylabel('intensity')
if j ==nComponents
    xlabel('measurement occasion');
end
if j==1
    title('Example time series');
    legend('PA','NA')
end
set(gca,'fontsize', fontsize)

end

% save plot
save('results/componentSave.mat','COEFFB','order','nameCompsOrdered')
saveas(gcf,'plots/plotPca.pdf')
saveas(gcf,'plots/plotPca.emf')

% save table
for i = 1:length(nameCompsOrdered)
    nameCompsOrderedX{i}=strrep(nameCompsOrdered{i},'\','');
    nameCompsOrderedX{i}=strrep(nameCompsOrderedX{i},' ','');
    nameCompsOrderedX{i}=strrep(nameCompsOrderedX{i},'*','R');
    nameCompsOrderedX{i}=strrep(nameCompsOrderedX{i},'{','');
    nameCompsOrderedX{i}=strrep(nameCompsOrderedX{i},'}','');
end
COEFFBTable=array2table(LOADINGROTOrder);
COEFFBTable.Properties.RowNames=namesX;
COEFFBTable.Properties.VariableNames=nameCompsOrderedX;
filename = 'results/PCAComponents.xlsx';
xlswrite(filename, {'Components'},1,'B1')
writetable(COEFFBTable,filename,'Sheet',1,'Range','A2','WriteRowNames',true);
sum(EXPLAINED(1:nComponents));

