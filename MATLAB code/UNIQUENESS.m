close all
clear all
clc

load('results/initAll.mat')

% put all statistics in one matrix
SLong=[];
idDatasetLong=[];
for i =1:length(SAll)
    SLong=[SLong;SAll{i}];
    idDatasetLong=[idDatasetLong;ones(length(SAll{i}),1)*i];
end


% Compute uniqueness
for i = 1:size(SLong,2);
    mdlM=fitlm(SLong(:,1:2),SLong(:,i));
    mdlSD=fitlm(SLong(:,1:4),SLong(:,i));
    RS(1,i)=mdlM.Rsquared.Ordinary; % explained by M
    RS(2,i)=mdlSD.Rsquared.Ordinary; % explained by SD
end

% only names after M and SD are interesting
for i =1:length(names)-4
    namesNOMSD{i}=names{i+4}
end

% create figure;
alphaFigure=1;
figure('position',[0 0 500*alphaFigure 500*alphaFigure])
rotateAngle=90;
co=get(gca,'colororder');
b=bar(RS(1:2,5:end)')
for i = 1:2
    b(i).FaceColor = co(i+1,:)
end
set(gca,'xtick',[1:(length(namesNOMSD))],'xticklabel',namesNOMSD)
xtickangle(rotateAngle)
ylabel('R^2')
axis square
axis ([0 length(namesNOMSD)+1 0 1])
legend('explained by M','explained by M and S');
saveas(gcf,'plots/Uniqueness.emf')

% save in table
tableU=array2table(RS);
tableU.Properties.VariableNames=namesX;
tableU.Properties.RowNames={'Explained by M','Explained by M and SD'};
add=1;
filename = 'results/Uniqueness.xlsx';
xlswrite(filename, {'R2'},1,['A' num2str(add)])
writetable(tableU,filename,'Sheet',1,'Range',['A' num2str(add+1)],'WriteRowNames',true)
