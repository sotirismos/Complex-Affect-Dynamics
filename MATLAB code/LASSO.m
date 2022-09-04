%%

close all
clear all
clc


%%%%%%%%%%% DECLARATIONS %%%%%%%%%%%%%%%%%%%%%%%%
namesTypesR={'Lasso results psychological wellbeing','Lasso results depressive symptoms','Lasso results life satisfaction'};
 
strD={'CESD','QIDS','PHQ','BDI'};
 strB={'ADPIV_BPD','PAIbor'};
 strSWL={'SWL'};

load('results/initAll.mat')

meanSets=[1 2];  % sets only mean
meanSDSets=[1 2 3 4]; % sets mean and sd

% load matlab functions
addpath(genpath('extraFiles'))


%%%%%%%%%%%%%%%%%%%%%%%%% REGRESSION LASSO PREDICTION %%%%%%%%%%%%%%%%%%


add=1;
addCluster=1;
% loop over data sets
for iDataset=1:length(dataStrAll)
    % loop over regression outcomes
    if length(RAll{iDataset}>0)
        for j=1:(size(RAll{iDataset},2)-1)
            clear EpredL EpredLM EpredLMSD choiceS YpredL YpredLM YpredLMSD
            
            %  create names
            nameRow{add}=[dataStrAll{iDataset} ': ' Rnames{iDataset,j}];
            nameRowFile{add}=[dataStrAll{iDataset} '_' Rnames{iDataset,j}];
            
            % add correct statistics to correct participants
            idNoNan=find(isnan(RAll{iDataset}(:,j+1))==0);
            Y=RAll{iDataset}(idNoNan,j+1); % outcome variable           
            S=SAll{iDataset}(idNoNan,:); % predictor variables
            
            % initialize choice
            for k =1:3
                RCHOICE{k}(add,:)=zeros(1,size(S,2));
            end
            
            % outer loop over Y
            for l=1:length(Y)                
                idxUse=[1:l-1 l+1:length(Y)];  % outer training set
                idxPred=l; % outer test set
                Yuse=Y(idxUse,1); %outer training set outcome
                Suse=S(idxUse,:); % outer training set predictor
                
                % load result of inner cross-validation to train LASSO if
                % already done, else do cross validation
                filename=['results/LASSO/Regression' nameRowFile{add} 'l' num2str(l) '.mat']; 
                if(~exist(filename))
                    [B,FitInfo] = lasso(Suse,Yuse,'CV',20); % all variables
                    [BM,FitInfoM] = lasso(Suse(:,meanSets),Yuse,'CV',20); % only mean
                    [BMSD,FitInfoMSD] = lasso(Suse(:,meanSDSets),Yuse,'CV',20); % mean and sd
                    save(filename,'B','BM','BMSD','FitInfo','FitInfoM','FitInfoMSD');
                else
                    load(filename);
                end
                
                percentOk=(iDataset-1)/length(dataStrAll)*100+...
                    (j-1)/(size(RAll{iDataset},2)-1)/length(dataStrAll)*100+...
                    l/length(Y)/(size(RAll{iDataset},2)-1)/length(dataStrAll)*100;
                
                disp([num2str(percentOk) '% Completed'])

                % choose best beta according to cross-validation for all variables  
                beta=[FitInfo.Intercept(FitInfo.IndexMinMSE);B(:,FitInfo.IndexMinMSE)];
                RCHOICE{1}(add,:)=RCHOICE{1}(add,:)+abs(sign(beta(2:end)'))/length(Y);
                YpredL(l,1)=[1 S(l,:)]*beta;
                EpredL(l,1)=(YpredL(l,1)-Y(l))^2;
                
                % choose best beta if only mean
                beta=[FitInfoM.Intercept(FitInfoM.IndexMinMSE);BM(:,FitInfoM.IndexMinMSE)];
                RCHOICE{2}(add,meanSets)=RCHOICE{2}(add,meanSets)+abs(sign(beta(2:end)'))/length(Y);
                YpredLM(l,1)=[1 S(l,meanSets)]*beta;
                EpredLM(l,1)=(YpredLM(l,1)-Y(l))^2;
                
                % choose best beta if mean and sd
                beta=[FitInfoMSD.Intercept(FitInfoMSD.IndexMinMSE);BMSD(:,FitInfoMSD.IndexMinMSE)];
                RCHOICE{3}(add,meanSDSets)=RCHOICE{3}(add,meanSDSets)+abs(sign(beta(2:end)'))/length(Y);
                YpredLMSD(l,1)=[1 S(l,meanSDSets)]*beta;
                EpredLMSD(l,1)=(YpredLMSD(l,1)-Y(l))^2;

                % predicted R2
                RLOO(add,:)=1-[mean(EpredL(1:l)) mean(EpredLM(1:l)) mean(EpredLMSD(1:l))]./var(Y);
                              
%                 [iDataset j l] % count where you are
                addCluster=addCluster+1;
            end  % do LOO
            NN(add,1)=length(Y);
            add=add+1;
        end
    end
end
disp(['100% Completed'])


% split in outcomes
clear typeOutcome 
add=1;
 for i =1:size(Rnames,1)
     for j =1:size(Rnames,2)
         if length(Rnames{i,j})>0
             found=0;
             for k=1:length(strD)
                 if strcmp(strD{k},Rnames{i,j})==1
                     typeOutcome(add)=1;
                     found=1;
                 end
             end
              for k=1:length(strSWL)
                 if strcmp(strSWL{k},Rnames{i,j})==1
                     typeOutcome(add)=2;
                     found=1;
                 end
              end
              if found==0
                  error('type not found')
              end
              add=add+1;
         end
     end
 end
 

 %%%%%%%%%%%%% CREATE PLOT AND SAVE RESULTS TO TABLES %%%%%%%%%%%%%%%%%%%%%%%%%%
alphaFigure=1;
figure('position',[0 0 800*alphaFigure 500*alphaFigure])

% loop over outcomes
for t=1:(max(typeOutcome)+1)
    
    if t==1
        idType=1:length(typeOutcome) % first time take everything
    else
        idType=find(typeOutcome==t-1)
    end
    
    % weighted average
    MRLOO(t,:)=(RLOO(idType,:)'*NN(idType)/sum(NN(idType)))';
%     SLOO(t,:)=sqrt(((RLOO(idType,:)-repmat(MRLOO(t,:),length(idType),1)).^2)'*NN(idType)/sum(NN(idType)))/sqrt(length(idType))
    
    % plot chosen variables
    subplot(2,2,t)    
    chosen=(RCHOICE{1}(idType,:)'*NN(idType)/sum(NN(idType)))'
    co=get(gca,'colororder');
    b2=bar(chosen);
    for i = 1:1
        b2(i).FaceColor = co(i,:);
    end
    set(gca,'xtick',[1:length(names)],'xticklabel',names)
    xtickangle(90)
    ylabel('% chosen')
    title(namesTypesR{t})
    
    keep2(t,:)=chosen;   
end

T=array2table(MRLOO);
T.Properties.VariableNames={'All','M','MSD'};
T.Properties.RowNames={'All','Depression','SWL'}

T2=array2table(keep2(1,:));
T2.Properties.VariableNames=namesX;
T2.Properties.RowNames={'percentageChosen'}

filename = 'results/LassoAnalysis.xlsx';
xlswrite(filename, {'Lasso results'},1,['B1' ])
writetable(T,filename,'Sheet',1,'Range',['A1' ],'WriteRowNames',true)

xlswrite(filename, {'Lasso results percentage variable chosen'},1,['B7' ])
writetable(T2,filename,'Sheet',1,'Range',['A8' ],'WriteRowNames',true)

%%

saveas(gcf,['plots/lassoResults.emf'])
h=gcf;
set(h,'PaperOrientation','landscape');
print(gcf, '-dpdf', ['plots/lassoResults.pdf'])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LASSO FOR CLASSIFICATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% NO EXAMPLE CLASSIFICATION OUTCOMES
% 
% close all
% clear all
% clc
% 
% load('results/initAll.mat')
% 
% 
% meanSets=[1 2];
% meanSDSets=[1 2 3 4];
% 
% 
% % classification
% add=1;
% addTest=1;
% % loop over data sets
% for iDataset=1:length(dataStrAll)
%     % loop over classification outcomes
%     if length(CAll{iDataset})>0
%         for j=1:(size(CAll{iDataset},2)-1)
%             clear EpredLC EpredLMC EpredLMSDC choiceSC YpredLC YpredLMC YpredLMSDC
%             idNoNan=find(isnan(CAll{iDataset}(:,j+1))==0);
%             Y=CAll{iDataset}(idNoNan,j+1);
%             S=SAll{iDataset}(idNoNan,:);           
%             LengthParts=[length(find(Y==0)), length(find(Y==1))]/length(Y);        
%             
%             % only use data set if enough participants per class
%             if min(LengthParts)>0.05 & strcmp(Cnames{iDataset,j},'SCID_MDD_BPD') == 0
%                 N(add,1)=length(Y);
%                 for k =1:3
%                     CCHOICE{k}(add,:)=zeros(1,size(S,2));
%                 end
%                 
%                 % name of rows
%                 nameRowC{add}=[dataStrAll{iDataset} ': ' Cnames{iDataset,j}];
%                 nameRowFileC{add}=[dataStrAll{iDataset} '_' Cnames{iDataset,j}];
%                 idNoNan=find(isnan(CAll{iDataset}(:,j+1))==0);    
%                 
%                 % outer loop
%                 for l=1:length(Y)
%                     idxUse=[1:l-1 l+1:length(Y)]; % idx of training data
%                     idxPred=l;
%                     Yuse=Y(idxUse,1);
%                     Suse=S(idxUse,:);
%                     
%                     % if file exist load, otherwise do cross-validation
%                     filename=['results/LASSO/Classification' nameRowFileC{add} 'l' num2str(l) '.mat']; 
%                     if(~exist(filename))
%                         [B,FitInfo] = lassoglm(Suse,Yuse,'binomial','CV',20);
%                         [BM,FitInfoM] = lassoglm(Suse(:,meanSets),Yuse,'binomial','CV',20);
%                         [BMSD,FitInfoMSD] = lassoglm(Suse(:,meanSDSets),Yuse,'binomial','CV',20);
%                         save(filename,'B','BM','BMSD','FitInfo','FitInfoM','FitInfoMSD');
%                     else
%                         load(filename);
%                     end
% 
%                     % results for all variables
%                     beta=[FitInfo.Intercept(FitInfo.IndexMinDeviance);B(:,FitInfo.IndexMinDeviance)];
%                     CCHOICE{1}(add,:)=CCHOICE{1}(add,:)+abs(sign(beta(2:end)'))/length(Y);
%                     YpredLC(l,:)=mnrval(beta,S(l,:));
%                     EpredLC(l,1)=(round(YpredLC(l,1))-Y(l))^2;
% 
%                     % results for only M
%                     beta=[FitInfoM.Intercept(FitInfoM.IndexMinDeviance);BM(:,FitInfoM.IndexMinDeviance)];
%                     CCHOICE{2}(add,meanSets)=CCHOICE{2}(add,meanSets)+abs(sign(beta(2:end)'))/length(Y);
%                     YpredLMC(l,:)=mnrval(beta,S(l,meanSets));
%                     EpredLMC(l,1)=(round(YpredLMC(l,1))-Y(l))^2;
% 
%                     % results for MSD
%                     beta=[FitInfoMSD.Intercept(FitInfoMSD.IndexMinDeviance);BMSD(:,FitInfoMSD.IndexMinDeviance)];
%                     CCHOICE{3}(add,meanSDSets)=CCHOICE{3}(add,meanSDSets)+abs(sign(beta(2:end)'))/length(Y);
%                     YpredLMSDC(l,:)=mnrval(beta,S(l,meanSDSets));
%                     EpredLMSDC(l,1)=(round(YpredLMSDC(l,1))-Y(l))^2;
% 
%                     % percentage correct
%                     CLOO(add,:)=1-[mean(EpredLC(1:l)) mean(EpredLMC(1:l)) mean(EpredLMSDC(1:l))];
% 
%                      percentOk=(iDataset-1)/length(dataStrAll)*100+...
%                     (j-1)/(size(CAll{iDataset},2)-1)/length(dataStrAll)*100+...
%                     l/length(Y)/(size(CAll{iDataset},2)-1)/length(dataStrAll)*100;
%                 
%                 disp([num2str(percentOk) '% Completed'])
%                 addTest=addTest+1;
%                 end  % do LOO
%                 NNC(add,1)=length(Y);
%                 add=add+1;
%             end
%         end
%     end
%   
% end
% disp(['100% Completed'])
% % plot results
% t=1
% idType=1:length(NNC);
% MRLOO(t,:)=(CLOO(idType,:)'*NNC(idType)/sum(NNC(idType)))'
% SLOO(t,:)=sqrt(((CLOO(idType,:)-repmat(MRLOO(t,:),length(idType),1)).^2)'*NNC(idType)/sum(NNC(idType)))/sqrt(length(idType))
% 
%     
% chosen=(CCHOICE{1}(idType,:)'*NNC(idType)/sum(NNC(idType)))'
% co=get(gca,'colororder');
% b2=bar(chosen);
% for i = 1:1
%     b2(i).FaceColor = co(i,:);
% end
% set(gca,'xtick',[1:length(names)],'xticklabel',names)
% xtickangle(90)
% ylabel('%')
% %     title('Variable Chosen IF all variable Lasso is best Lasso')
%     
% % variable chosen 
% keep2(t,:)=chosen
% 
% % lasso chosen
% MRLOO