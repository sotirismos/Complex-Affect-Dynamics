%% Individual data set regression analyais and putting regression outcomes in mixed model format
close all
clear all
clc

%%%%%%%%%%%%%%%%%%%%%%% DECLARATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% outcomes per type
strD={'CESD','QIDS','PHQ','BDI'}; % depression 
% strB={'ADPIV_BPD','PAIbor'}; % BPD In this online package we don't have
% BPD outcomes
strSWL={'SWL'}; % Satisfaction with life

% load data
load('results/initAll.mat')

% load matlab functions
addpath(genpath('extraFiles'))


%%%%%%%%%%%%%% COMPUTE REGRESSION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% no intercepts will be used (dependent and independent variables are standardized)
SLong=[];
idDatasetLong=[];
for i =1:length(SAll)
    SLong=[SLong;SAll{i}];
    idDatasetLong=[idDatasetLong;ones(length(SAll{i}),1)*i];
end


% regression
add=1;
dummyOutcome=[];
% loop over data sets
for iDataset=1:length(dataStrAll)
    % loop over regression outcomes in data set
    if length(RAll{iDataset})>0
        for j=1:(size(RAll{iDataset},2)-1)
            % find the right statistics for the right participants
            idNoNan=find(isnan(RAll{iDataset}(:,j+1))==0);
            Y=zscore(RAll{iDataset}(idNoNan,j+1)); % standardize outcome           
            Stemp=SAll{iDataset}(idNoNan,:);
                        
            % standardize
            S=zeros(size(Stemp,1),size(Stemp,2));
            for k =1:size(Stemp,2)
                S(:,k)=(Stemp(:,k)-nanmean(Stemp(:,k)))/nanstd(Stemp(:,k));
            end

            % mean and sd statistics
            SetM=[1 2]; % MEAN COLUMNS
            SetSD=[1 2 3 4]; % MEAN AND SD COLUMNS
            SM=[S(:,SetM)]; % mean
            SSD=[S(:,SetSD)]; % mean and sd
            
            % model with means    
            mdlM=fitlm(SM,Y);
            R2M(add)=mdlM.Rsquared.Ordinary;

            % model with SD and M
            mdlSD=fitlm(SSD,Y);
            R2SD(add)=mdlSD.Rsquared.Ordinary;

            % compute models for solo variables
            for k=1:size(S,2)
                SSolo=[S(:,k)];
                mdlSolo=fitlm(SSolo,Y);
                R2Solo(add,k)=mdlSolo.Rsquared.Ordinary;
                pSolo(add,k)=mdlSolo.Coefficients.pValue(end);
                BetaSolo(add,k)=mdlSolo.Coefficients.Estimate(end);
                SESolo(add,k)=mdlSolo.Coefficients.SE(end);
            end

            % compute models for Over Mean
            for k=3:size(S,2)
                SOverM=[SM S(:,k)];
                mdlOverM=fitlm(SOverM,Y);
                BetaOverM(add,k)=mdlOverM.Coefficients.Estimate(end);
                SESOverM(add,k)=mdlOverM.Coefficients.SE(end);
                R2WithM(add,k-2)=mdlOverM.Rsquared.Ordinary;
                R2OverM(add,k-2)=R2WithM(add,k-2)-R2M(add);
                pOverM(add,k-2)=mdlOverM.Coefficients.pValue(end);              
            end

            % compute models for Over SD
            for k=5:size(S,2)
                SOverMSD=[SSD S(:,k)];
                mdlOverMSD=fitlm(SOverMSD,Y);
                BetaOverMSD(add,k)=mdlOverMSD.Coefficients.Estimate(end);
                SESOverMSD(add,k)=mdlOverMSD.Coefficients.SE(end);
                R2WithMSD(add,k-4)=mdlOverMSD.Rsquared.Ordinary;
                R2OverMSD(add,k-4)=R2WithMSD(add,k-4)-R2SD(add);
                pOverMSD(add,k-4)=mdlOverMSD.Coefficients.pValue(end);
            end
            
            nameRow{add}=[dataStrPaper{iDataset} ': ' Rnames{iDataset,j}]; % give name to row
             
            % LOOK FOR TYPE OUTCOME
            dummyOutcomeTemp=zeros(length(Y),2);
            found=0;
            for k=1:length(strD)
                if strcmp(strD{k},Rnames{iDataset,j})==1
                    typeOutcomeR(add)=1;
                    dummyOutcomeTemp(:,1)=1;
                    found=1;
                end
            end
%             for k=1:length(strB)
%                 if strcmp(strB{k},Rnames{iDataset,j})==1
%                     typeOutcomeR(add)=2;
%                     dummyOutcomeTemp(:,2)=1;
%                     found=1;
%                 end
%             end
            for k=1:length(strSWL)
                if strcmp(strSWL{k},Rnames{iDataset,j})==1
                    typeOutcomeR(add)=2;
                    dummyOutcomeTemp(:,2)=1;
                    found=1;
                end
            end
            if found==0
                error('type not found')
            end            
            dummyOutcome=[dummyOutcome;dummyOutcomeTemp];
            
            
            % create mixed model data
            if typeOutcomeR(add)==2 % change sign satisfaction with life
                YMixedCell{add}=-1*Y;
            else
                YMixedCell{add}=Y;
            end   
            SMixedCell{add}=S;
            GDatasetCell{add}=ones(length(Y),1)*iDataset;
            
            NN(add,1)=size(S,1);  % number of participants
            add=add+1;
        end
    end 
    disp([num2str(iDataset/length(dataStrAll)*100) '% Completed'])
end

% put mixed model data in long vectors
SMixed=[];
YMixed=[];
GMixed=[];
GDataset=[];
for i = 1:length(SMixedCell)
    SMixed=[SMixed;SMixedCell{i}];
    YMixed=[YMixed;YMixedCell{i}];
    GMixed=[GMixed;ones(length(YMixedCell{i}),1)*i];
    GDataset=[GDataset;GDatasetCell{i}];
end


% weighted mean R2 for individual regressions
for t = 1:4
    % if type = 1, all outcomes, else specific outcomes
    if t==1
        idType=1:length(typeOutcomeR);
    else
        idType=find(typeOutcomeR==(t-1));
    end

    % compute weighted averages
    R2Seperate{t}=nan(3,size(R2Solo,2));
    R2Seperate{t}(1,:)=R2Solo(idType,:)'*NN(idType)/sum(NN(idType));
    R2Seperate{t}(2,3:end)=R2OverM(idType,:)'*NN(idType)/sum(NN(idType));
    R2Seperate{t}(3,5:end)=R2OverMSD(idType,:)'*NN(idType)/sum(NN(idType));
    
    % weighed standard deviation (does not include within data set error)
    R2SeperateS{t}=nan(3,size(R2Solo,2));
    R2SeperateS{t}(1,:)=sqrt((((R2Solo(idType,:)-repmat(R2Seperate{t}(1,:),length(idType),1)).^2)'*NN(idType))/sum(NN(idType)))/sqrt(length(idType));
    R2SeperateS{t}(2,3:end)=sqrt((((R2OverM(idType,:)-repmat(R2Seperate{t}(2,3:end),length(idType),1)).^2)'*NN(idType))/sum(NN(idType)))/sqrt(length(idType));
    R2SeperateS{t}(3,5:end)=sqrt((((R2OverMSD(idType,:)-repmat(R2Seperate{t}(3,5:end),length(idType),1)).^2)'*NN(idType))/sum(NN(idType)))/sqrt(length(idType));
end

save('results/singleFreqAnalysis.mat')

%% Mixed Model Analysis Normal and Weighted (against Heteroscedastisity) and test assumptions
close all
clear all
clc

% load matlab functions
addpath(genpath('extraFiles'))

% load data
load('results/singleFreqAnalysis.mat')

% path where to save assumption test plots
basicPath='appendix';

% create empty arrays
for i =1:4
    pMeta{i}=NaN(3,size(SMixed,2)); % pvalues meta
    R2Meta{i}=NaN(3,size(SMixed,2)); % R2 values meta
    pMetaNoH{i}=NaN(3,size(SMixed,2)); % pvalues weighted analysis
end

% create empty arrays for assumption tests
% with and without type outcome included
for i =1:2
    BPKeep{i}=NaN(3,size(SMixed,2)); % Breusch–Pagan values for Heteroscedastisity test
    BPKeepPost{i}=NaN(3,size(SMixed,2)); % Breusch–Pagan values for Heteroscedastisity test after weighted analysis
    ksMixed{i}=NaN(3,size(SMixed,2)); % for kolmogorov smirnov test
end

% strings to use in creation latex file (Meta analytic results appendix and
% test assumptions appendix)
typeNames={'psychological well-being','depression','borderline','life satisfaction'};
namesLatex=names;
namesLatex{11}=['\' namesLatex{11}];
stringLatexAssumptions=''; % empty string to fill with appendix text
stringLatexTextModel='\\section{Regression}';
stringLatexTableBegin=[char(10) '\\begin{table}[H]' char(10) ...
'\\begin{center}' char(10) ...
'\\begin{tabular}{l r r r r r r r}' char(10) ...
'& estimate & SE & tstat & DF & pval & 95\\%% conf int lower bound  & 95\\%% conf int upper bound \\\\' char(10) ...
'\\hline' char(10)];
stringLatexTableEnd='\\end{tabular} \\end{center} \\end{table} \\clearpage';
dummyNames={'dummy_D','dummy_SWL'};


addFig=1; % counter for figures

% loop over models (solo, over M and over MSD)
for m=1:3       
    % basis: over what do we predict
    if m==1 % solo
        basis=[];
        stringLatexAssumptions=[stringLatexAssumptions '\\section{Solo variables}' char(10)];
        stringLatexTextModel=[ stringLatexTextModel '\\subsection{Solo variables}' char(10)];
    elseif m==2 % over M
        basis=[1 2];
        stringLatexAssumptions=[stringLatexAssumptions '\\section{Over M}' char(10)];
        stringLatexTextModel=[stringLatexTextModel '\\subsection{Over M}' char(10)];
    else % over M and SD
        basis=[1 2 3 4];
        stringLatexAssumptions=[stringLatexAssumptions '\\section{Over M and SD}' char(10)];
        stringLatexTextModel=[stringLatexTextModel '\\subsection{Over M and SD}' char(10)];
    end    

    % create empty tables
    for t=1:(max(typeOutcomeR)+1)
        tableLatex{t}=['\\subsubsection{Overview ' typeNames{t} '}' char(10) stringLatexTableBegin];
    end
    modelLatexTemp=''; % empty latex file
    
    % create the basis over which we want to predict
    if length(basis)>0
        % without type outcome
        X=[ones(size(SMixed,1),1) SMixed(:,basis)]; % intercept and statistics
        mdl=fitlmematrix(X,YMixed,X,GMixed);
        R2Basis=R2Mixed(mdl,dummyOutcome);

        % with type outcome: create variables * dummy for type
        XFixed=[];
        for k=1:size(X,2)
             XFixed=[XFixed repmat(X(:,k),1,size(dummyOutcome,2)).*dummyOutcome];
        end
        mdl=fitlmematrix(XFixed,YMixed,X,GMixed);
        [~,R2BasisType]=R2Mixed(mdl,dummyOutcome);
    end

    % loop over all dynamics not in basis
    for j =(length(basis)+1):size(SMixed,2)
        
        % INDEPENDENT OF TYPE OUTCOME
        
        % create names fixed effects and random effects
        clear FixedNames
        FixedNames{1}='Intercept';
        for ib=1:length(basis)
            FixedNames{ib+1}=namesX{ib};
        end
        FixedNames{length(basis)+2}=namesX{j};
        GroupName{1}='Questionnaire';
        clear RandomNamestemp
        for ib=1:length(FixedNames)
            RandomNamesTemp{ib}=[FixedNames{ib} '_'];
        end
        RandomNames{1}=RandomNamesTemp;
        
        % estimate model independent of type outcome
        X=[ones(size(SMixed,1),1) SMixed(:,basis) SMixed(:,j)];
        mdl=fitlmematrix(X,YMixed,X,GMixed,'FixedEffectPredictors',FixedNames,'RandomEffectPredictors',RandomNames,'RandomEffectGroups',GroupName);  
        
        % write all information in latex file
        tableLatex{1}=[tableLatex{1} '$' namesLatex{j} '$' '&' num2str(mdl.Coefficients.Estimate(end),4) '&' num2str(mdl.Coefficients.SE(end),4) '&'...
            num2str(mdl.Coefficients.tStat(end),4) '&' num2str(mdl.Coefficients.DF(end),4) '&' num2str(mdl.Coefficients.pValue(end),4)...
            '&' num2str(mdl.Coefficients.Lower(end),4) '&' num2str(mdl.Coefficients.Upper(end),4)  '\\\\' char(10)];
        modelLatexTemp=[modelLatexTemp '\\subsubsection{$' namesLatex{j} '$: general psychological well-being}' char(10)];
        stringModelTemp=evalc('disp(mdl)');
        stringModelTemp2 = strrep(stringModelTemp,'%','%%');
        stringModelTemp3 = strrep(stringModelTemp2,'<strong>','');
        stringModelTemp4 = strrep(stringModelTemp3,'</strong>','');
        modelLatexTemp=[modelLatexTemp '\\begin{lstlisting}'  stringModelTemp4 char(10) '\n \\end{lstlisting} \\clearpage'];

        % extract p value
        pMeta{1}(m,j)=mdl.Coefficients.pValue(end);        
          
        

        
        % compute R2 (or additional R2)
        R2MixedTemp=R2Mixed(mdl,dummyOutcome);
        if length(basis>0)
            R2Meta{1}(m,j)=R2MixedTemp-R2Basis;
        else
            R2Meta{1}(m,j)=R2MixedTemp;
        end
        
        % TEST ASSUMPTIONS
        % normality
        resids=residuals(mdl,'ResidualType','Standardized'); % extract standardized residuals
        [~,ksMixed{1}(m,j)]=kstest(resids); % compute kolmogorov smirnov
               
        % heteroscedasticity
        % applied linear statistical models fifth edition (kutner et al)
        % page 426 on top        
        resid=mdl.residuals; % extract residuals        
        BPKeep{1}(m,j)=BPTest(resid,X); % breush pagan test
        residualsa=abs(resid); % extract absolute value residuals
        mdlR=fitlmematrix(X,residualsa,X,GMixed); % fit model on |residuals|
        W=1./(mdlR.fitted.^2); % weightes are inverse of fitted variance
        mdlNoH=fitlmematrix(X,YMixed,X,GMixed,'Weights',W); % new mixed model with weights             
        pMetaNoH{1}(m,j)=mdlNoH.Coefficients.pValue(end);   % weighted pvalue
        residW=mdlNoH.residuals.*sqrt(W); % new residuals
        BPKeepPost{1}(m,j)=BPTest(residW,X); % breush pagan test for new residuals
            
        % WITH TYPE OUTCOME
        % create new fixed effects (and fixed effect names) using dummy for type outcome
        XFixed=[]; 
        addR=1;
        clear FixedNamesTemp
        for k=1:size(X,2)
             XFixed=[XFixed repmat(X(:,k),1,size(dummyOutcome,2)).*dummyOutcome];
             for ll=1:length(dummyNames)
                 FixedNamesTemp{addR}=[FixedNames{k} '_X_' dummyNames{ll}];
                 addR=addR+1;
             end
        end
        FixedNames=FixedNamesTemp;

        
        % estimate new model
        mdlO=fitlmematrix(XFixed,YMixed,X,GMixed,'FixedEffectPredictors',FixedNames,'RandomEffectPredictors',RandomNames,'RandomEffectGroups',GroupName);        
        [~,R2MixedTempO]=R2Mixed(mdlO,dummyOutcome);              

        % write everything in latex file
        modelLatexTemp=[modelLatexTemp '\\subsubsection{$' namesLatex{j} '$: including type outcomes}' char(10)];
        stringModelTemp=evalc('disp(mdlO)');
        stringModelTemp2 = strrep(stringModelTemp,'%','%%');
        stringModelTemp3 = strrep(stringModelTemp2,'<strong>','');
        stringModelTemp4 = strrep(stringModelTemp3,'</strong>','');
        modelLatexTemp=[modelLatexTemp '\\begin{lstlisting}'  stringModelTemp4 char(10) '\n \\end{lstlisting} \\clearpage'];
        for t=1:max(typeOutcomeR)
                    tableLatex{t+1}=[tableLatex{t+1} '$' namesLatex{j} '$' '&' num2str(mdlO.Coefficients.Estimate(end-3+t),4) '&' num2str(mdlO.Coefficients.SE(end-3+t),4) '&'...
            num2str(mdlO.Coefficients.tStat(end-3+t),4) '&' num2str(mdlO.Coefficients.DF(end-3+t),4) '&' num2str(mdlO.Coefficients.pValue(end-3+t),4)...
            '&' num2str(mdlO.Coefficients.Lower(end-3+t),4) '&' num2str(mdlO.Coefficients.Upper(end-3+t),4)  '\\\\' char(10)];      
        end       
        
        % extract R2
        for t=1:max(typeOutcomeR)
            pMeta{t+1}(m,j)=mdlO.Coefficients.pValue(end-max(typeOutcomeR)+t);  % end-2, end-1 and end
            if length(basis)>0
                R2Meta{t+1}(m,j)=R2MixedTempO(t)-R2BasisType(t);
            else
                R2Meta{t+1}(m,j)=R2MixedTempO(t);
            end
        end
               
        
        % TEST ASSUMPTIONS WITH TYPE OUTCOME
        % normality
        residsT=residuals(mdlO,'ResidualType','Standardized');
        [~,ksMixed{2}(m,j)]=kstest(residsT);
        
        % heteroscedasticity
        residT=mdlO.residuals; % extract residuals
        BPKeep{2}(m,j)=BPTest(residT,X); % breush pagan test
        residualsa=abs(residT); % extract absolute value residuals
        mdlR=fitlmematrix(XFixed,residualsa,X,GMixed);% fit model on |residuals|
        W=1./(mdlR.fitted.^2); % weightes are inverse of fitted variance
        mdlNoH=fitlmematrix(XFixed,YMixed,X,GMixed,'Weights',W); % new mixed model with weights      
        residWT=mdlNoH.residuals.*sqrt(W); % new residuals
        BPKeepPost{2}(m,j)=BPTest(residWT,XFixed); % breush pagan test for new residuals   
        
        % p values weighted regression
        for t=1:max(typeOutcomeR)
            pMetaNoH{t+1}(m,j)=mdlNoH.Coefficients.pValue(end-max(typeOutcomeR)+t);
        end
        
        % CREATE PICTURES FOR ASSUMPTION APPENDIX
        % qqplot, hist plot, errors versus fitted, errors vs dynamic
        % measure
        
        stringLatexAssumptions=[stringLatexAssumptions char(10) '\\subsection{$' namesLatex{j} '$}' char(10) ];               
        % without type
        fig=figure('position',[0 0 800 1000]);
        stringLatexAssumptions=[stringLatexAssumptions char(10) '\\subsubsection{Model for general psychological well-being}' char(10) figureString(addFig,0.95,'plotH') char(10)];       
        subplot(3,2,1)
        qqplot(resids);
        title(['Q-Q plot,  ' char(10) 'K-test p-value: ' num2str(ksMixed{1}(m,j))]);      
        subplot(3,2,2)
        histfit(resids);
        title(['histogram,  ' char(10) 'K-test p-value: ' num2str(ksMixed{1}(m,j))]);  
        xlabel('residuals');
        ylabel('frequency')
        legend('bin','fitted normal')
        subplot(3,2,3)
        plot(mdl.fitted,resid,'.');
        xlabel('fitted y');
        ylabel('residuals');
        title(['Heteroscedasticity plot,  ' char(10) 'BP-test p-value: ' num2str(BPKeep{1}(m,j))]); 
        subplot(3,2,4)
        plot(X(:,end),resid,'.');
        xlabel(names{j});
        ylabel('residuals');
        title(['Heteroscedasticity plot,  ' char(10) 'BP-test p-value: ' num2str(BPKeep{1}(m,j))]);  
        subplot(3,2,5)
        plot(mdl.fitted,residW,'.');
        xlabel('fitted y');
        ylabel('residuals');
        title(['Heteroscedasticity plot weighted regression,  ' char(10) 'BP-test p-value: ' num2str(BPKeepPost{1}(m,j))]); 
        subplot(3,2,6)
        plot(X(:,end),residW,'.');
        xlabel(names{j});
        ylabel('residuals');
        title(['Heteroscedasticity plot weighted regression,  ' char(10) 'BP-test p-value: ' num2str(BPKeepPost{1}(m,j))]);  
        print([basicPath '\plotH\fig' num2str(addFig)],'-depsc')

        close(fig)
        addFig=addFig+1;
           
        
        % with type
        stringLatexAssumptions=[stringLatexAssumptions char(10) '\\subsubsection{Model for specific psychological well-being outcomes}' char(10) figureString(addFig,0.95,'plotH') char(10)];       
        fig=figure('position',[0 0 800 1000]);  
        subplot(3,2,1)
        qqplot(residsT);
        title(['Q-Q plot,  ' char(10) 'K-test p-value: ' num2str(ksMixed{2}(m,j))]);      
        subplot(3,2,2)
        histfit(residsT);
        title(['histogram,  ' char(10) 'K-test p-value: ' num2str(ksMixed{2}(m,j))]);  
        xlabel('residuals');
        ylabel('frequency')
        legend('bin','fitted normal')
        subplot(3,2,3)
        plot(mdl.fitted,residT,'.');
        xlabel('fitted y');
        ylabel('residuals');
        title(['Heteroscedasticity plot,  ' char(10) 'BP-test p-value: ' num2str(BPKeep{2}(m,j))]); 
        subplot(3,2,4)
        plot(X(:,end),residT,'.');
        xlabel(names{j});
        ylabel('residuals');
        title(['Heteroscedasticity plot,  ' char(10) 'BP-test p-value: ' num2str(BPKeep{2}(m,j))]);          
        subplot(3,2,5)
        plot(mdl.fitted,residWT,'.');
        xlabel('fitted y');
        ylabel('residuals');
        title(['Heteroscedasticity plot weighted regression,  ' char(10) 'BP-test p-value: ' num2str(BPKeepPost{2}(m,j))]); 
        subplot(3,2,6)
        plot(X(:,end),residWT,'.');
        xlabel(names{j});
        ylabel('residuals');
        title(['Heteroscedasticity plot weighted regression,  ' char(10) 'BP-test p-value: ' num2str(BPKeepPost{2}(m,j))]);   
        print([basicPath '\plotH\fig' num2str(addFig)],'-depsc')
        
        close(fig)
        addFig=addFig+1;
        

        disp([num2str(((m-1)*size(SMixed,2)+j)/(3*size(SMixed,2))*100) '% Completed'])
    end
    
    % add latex files together for meta analytic result appendix
    for t=1:(max(typeOutcomeR)+1)
        stringLatexTextModel=[stringLatexTextModel char(10) tableLatex{t} stringLatexTableEnd];
    end
    stringLatexTextModel=[stringLatexTextModel char(10) modelLatexTemp char(10)];
    
    
end


fid = fopen([basicPath '/rawVisual.tex'],'wt');
fprintf(fid, stringLatexAssumptions);
fclose(fid);

fid = fopen([basicPath '/rawModel.tex'],'wt');
fprintf(fid, stringLatexTextModel);
fclose(fid);

save('results/R2Pmeta.mat','R2Meta','pMeta','pMetaNoH','BPKeep','BPKeepPost','ksMixed');

%% compute predicted R2 for regression outcomes

%R2=1-MSE/var(Y), where MSE comes from cross-validation

close all
clear all
clc

% load data
load('results/singleFreqAnalysis.mat')


% load matlab functions
addpath(genpath('extraFiles'))


% create empty arrays
for i =1:3
    R2Predicted{i}=NaN(3,size(SMixed,2));
    R2PredictedSE{i}=NaN(3,size(SMixed,2));
end

% loop over models (solo, over M and over MSD)
for m=1:3       
    % basis: over what do we predict
    if m==1
        basis=[];
    elseif m==2
        basis=[1 2];
    else
        basis=[1 2 3 4];
    end    

    % create the basis
    if length(basis)>0
        % without type
        X=[ones(size(SMixed,1),1) SMixed(:,basis)];
        ErrorsBasics2=R2MixedCV(X,YMixed,X,GMixed);

        % with type
        XFixed=[];
        for k=1:size(X,2)
             XFixed=[XFixed repmat(X(:,k),1,size(dummyOutcome,2)).*dummyOutcome];
        end
        ErrorsBasicsType2=R2MixedCV(XFixed,YMixed,X,GMixed);
    end

    % go over all dynamics not in basis
    for j =(length(basis)+1):size(SMixed,2)
        
        % no type
        X=[ones(size(SMixed,1),1) SMixed(:,basis) SMixed(:,j)];
        Errors2=R2MixedCV(X,YMixed,X,GMixed);      
        if length(basis>0)
            R2Predicted{1}(m,j)=(1-mean(Errors2)/var(YMixed))-(1-mean(ErrorsBasics2)/var(YMixed)); %added R2 over basis
            R2PredictedSE{1}(m,j)=WithinBetweenSE((Errors2-ErrorsBasics2).^2,GMixed)/var(YMixed); % standard error of R2 : standard error of MSE divided by var(YMixed)
        else
            R2Predicted{1}(m,j)=1-mean(Errors2)/var(YMixed); % solo R2
            R2PredictedSE{1}(m,j)=WithinBetweenSE(Errors2,GMixed)/var(YMixed); % standard error of R2 : standard error of MSE divided by var(YMixed);
        end

        
        % with type
        XFixed=[];
        % create new fixed effects using dummy for type outcom
        for k=1:size(X,2)
             XFixed=[XFixed repmat(X(:,k),1,size(dummyOutcome,2)).*dummyOutcome];
        end
        ErrorsType2=R2MixedCV(XFixed,YMixed,X,GMixed);
        for t=1:max(typeOutcomeR)
            idTemp=find(dummyOutcome(:,t)==1);
            if length(basis)>0
                R2Predicted{t+1}(m,j)=(1-mean(Errors2(idTemp))/var(YMixed(idTemp)))-(1-mean(ErrorsBasics2(idTemp))/var(YMixed(idTemp)));
                R2PredictedSE{t+1}(m,j)=WithinBetweenSE(ErrorsType2(idTemp)-ErrorsBasicsType2(idTemp),GMixed(idTemp))/var(YMixed(idTemp));
            else
                R2Predicted{t+1}(m,j)=1-mean(Errors2(idTemp))/var(YMixed(idTemp));
                R2PredictedSE{t+1}(m,j)=WithinBetweenSE(ErrorsType2(idTemp),GMixed(idTemp))/var(YMixed(idTemp));
            end
        end
        
        disp([num2str(((m-1)*size(SMixed,2)+j)/(3*size(SMixed,2))*100) '% Completed'])
    end
end

save('results/R2Pred.mat','R2Predicted','R2PredictedSE');


%% Testing covariates: IMPOSSIBLE WITH ONLY TWO DATA SETS IN ONLINE PACKAGE
% 
% close all
% clear all
% clc
% 
% % load data
% load('results/singleFreqAnalysis.mat')
% 
% % reverse YMixed: look at coefficients to explain WellBeing (and not
% % depression)
% 
% YMixed=-YMixed;
% 
% % create empty cells to save p-values
% for m=1:3
%     pValAllCov{m}=NaN(length(COVDataset),size(SMixed,2)); % save all covariates
% end
% 
% % loop over covariates
% for covi=1:length(COVDataset)
%     
%     % loop over models (solo, over M and over MSD)
%     for m=1:3   
%         % basis: over what do we predict
%         if m==1
%             basis=[];
%         elseif m==2
%             basis=[1 2];
%         else
%             basis=[1 2 3 4];
%         end    
% 
%         % loop over all statistics for 1 covariate
%         lineCovi=NaN(1,size(SMixed,2));
%         parfor j =(length(basis)+1):size(SMixed,2)       
%             
%             % no covariates
%             Xpf=[ones(size(SMixed,1),1) SMixed(:,basis) SMixed(:,j)];           
%             
%             % with covariates for fixed effects
%             XFixed=[];            
%             for k=1:size(Xpf,2) % loop over variables                
%                  XFixed=[XFixed Xpf(:,k)]; % add variable
%                  XFixed=[XFixed Xpf(:,k).*COVDataset(GDataset,covi)]; % add interaction
%             end
%             mdlO=fitlmematrix(XFixed,YMixed,Xpf,GMixed); % estimate model
%             
%             lineCovi(j)=mdlO.Coefficients.pValue(end); % extract p value covariate effect
%             betaBase(j)=mdlO.Coefficients.Estimate(end-1); % extract beta base
%             betaExtra(j)=mdlO.Coefficients.Estimate(end); % extract interaction with covariate            
%         end
%         
%         % save in cells
%         pValAllCov{m}(covi,:)=lineCovi;
%         estAllBase{m}(covi,:)=betaBase;
%         betaAllExtra{m}(covi,:)=betaExtra;
%         
%         disp([num2str(((covi-1)*3+m)/(3*length(COVDataset))*100) '% Completed'])
%         
%     end
% end
% 
% % save everything
% save('results/covariateAnalysis.mat')
% 
% 
% %% extract results covariate analysis and execute false discovery rate
% close all
% clear all
% clc
% 
% % file where to save output
% filename = 'results/covSignificance.xlsx';
% 
% % load data
% load('results/covariateAnalysis.mat')
% 
% % alpha levels to test on
% alphas=[0.05 0.01 0.001];
% 
% % DO False Discovery Rate Benjamini–Hochberg procedure
% % (https://en.wikipedia.org/wiki/False_discovery_rate)
% % loop over alpha levels
% for a=1:length(alphas)
%     alpha=alphas(a);
%     for m =1:3 % do different FDR per model       
%         if m==1
%             P=pValAllCov{m}(:,1:end);
%         elseif m==2
%             P=pValAllCov{m}(:,3:end);
%         else
%             P=pValAllCov{m}(:,5:end);
%         end
%         [Ps]=sort(P(:)); % sort p values
%         kma=(1:length(Ps))'*alpha/length(Ps); 
%         temp=[Ps kma Ps-kma];
%         idOk=find(temp(:,3)<0); % reject 0 where kma>Ps
%         % save cut off p value
%         if length(idOk)>0
%             pOk(m,a)=Ps(idOk(end))*1.0001; 
%         else
%             pOk(m,a)=0;
%         end
%     end
% end
% 
% % loop over models (Solo, overM and overMSD)
% for m=1:3  
%     % loop over covariate matrices
%     for covi=1:length(COVDataset)     
%         % select suitable basis
%         if m==1
%             basis=[]; % for solo
%         elseif m==2
%             basis=[1 2]; % for M
%         else
%             basis=[1 2 3 4]; % for MSD
%         end  
%         
%         % loop over statistics
%         for j =(length(basis)+1):size(SMixed,2)  
%             
%                 covmin=min(COVDataset(unique(GDataset),covi)); % minimum covariate
%                 covmax=max(COVDataset(unique(GDataset),covi)); % maximum ovariate
%             
%                 b1min=estAllBase{m}(covi,j)+betaAllExtra{m}(covi,j)*covmin; % minumum coefficient
%                 b1max=estAllBase{m}(covi,j)+betaAllExtra{m}(covi,j)*covmax; % maximum coefficient
%                 
%                 % create string to write in excell: from minumum
%                 % coefficient to max
%                 StringCell{m}{covi,j+1}=[num2str(round(b1min,2)) ' to ' num2str(round(b1max,2))];
%                 
%                 % loop over FDR alpha levels
%                 for a=1:length(alphas)
%                     if pValAllCov{m}(covi,j)<pOk(m,a)
%                         StringCell{m}{covi,j+1}=[StringCell{m}{covi,j+1} '*']; % add star for each alpha level
%                     end
%                 end
%              
%                 % first column is minimum and maximum covariate
%                 StringCell{m}{covi,1}=[num2str(round(covmin,2)) ' to ' num2str(round(covmax,2))];
%         end
%     end
% end
% 
% % create column names
% colnamesCov{1}='Covariates_min_max';
% for i =1:length(namesX)
%     colnamesCov{i+1}=namesX{i};
% end
% 
% % write tables
% for m =1:3
%     T=cell2table(StringCell{m});
%     T.Properties.VariableNames=colnamesCov;
%     T.Properties.RowNames=namesCov;        
%     writetable(T,filename,'Sheet',1,'Range',['A' num2str((m-1)*(length(namesCov)+4)+1)],'WriteRowNames',true)
% end


%% SUBSAMPLING ANALYSIS: IMPOSSIBLE WITH ONLY 2 DATSETS

% 
% %%%%% CALCULATE NUMBER OF SUBSAMPLES%%%%%
% % load data
% load('results/singleFreqAnalysis.mat')
% 
% % look at all datasets
% uiG=unique(GDataset);
% for i =1:length(uiG)
%     L(i)=length(find(GDataset==uiG(i)));
% end
% 
% % total length of all observations together
% totalL=length(GDataset);
% useHalf=totalL/2;
% 
% % all possible subsets
% subsets=de2bi(0:(2^13-1));
% 
% % compute length of first subset
% for i = 1:length(subsets)
%     useIdx=find(subsets(i,:)==1);
%     NSetOne(i)=sum(L(useIdx));
% end
% 
% % get all subsets whith less then half of the observations but more then
% % 45%
% useSets=find(NSetOne<=useHalf & NSetOne/sum(L)>0.45);
% 
% % delete complementary data sets (with exactly half of the observations)
% adde=1;
% for i = 1:length(useSets)
%     ok=1;
%     for j =i+1:length(useSets)
%         tempTest=subsets(useSets(i),:)+subsets(useSets(j),:);  % subset one is 1-subset 2    
%         if max(tempTest)==1 & min(tempTest)==1
%             ok=0;
%         end
%     end
%     if ok==1
%         useSetsFinal(adde)=useSets(i);
%         adde=adde+1;
%     end
% end
% 
% 
% %%%%%%%%%%% DO SUBSAMPLES %%%%%%%%%%%%%%%%%
% for i =1:length(NSubsets)
%  % do on super computer because too slow
%  %     SubsampleCluster(num2str(i)); 
% end
% 
% cutoff=0.001;
% 
% %%%%% LOAD SUBSAMPLES %%%%%%%%%%%%%%%%
% pMetaAllSame=zeros(3,size(SMixed,2));
% for i =1:NSets
%     load(['results/subsample/mdl' num2str(i) '.mat']) % load subsamples
%     ptemp1=zeros(3,size(SMixed,2)); % empty array
%     ptemp2=zeros(3,size(SMixed,2)); % empty array
%     
%     ptemp1(find(pMetaTemp1<cutoff))=1; % set to 1 if significant
%     ptemp2(find(pMetaTemp2<cutoff))=1; % set to 1 if significant   
% 
%     same=1-abs(ptemp1-ptemp2); % set to 1 if both are same
%     pMetaAllSame=pMetaAllSame+same/NSets; % add to get proportion
% end
% 
% % create table
% T=array2table(round(pMetaAllSame,2));
% T.Properties.VariableNames=namesX;
% T.Properties.RowNames={'Solo','Over M','Over MSD'};
% filename = 'results/subsamples.xlsx';
% writetable(T,filename,'Sheet',1,'Range',['A1'],'WriteRowNames',true)
% 


%% BOOTSTRAP
close all
clear all
clc

B=2000; % number of bootstraps

%%%%%%%%%%% DO SUBSAMPLES %%%%%%%%%%%%%%%%%
for i =1:B
 % do on super computer because too slow
      BootstrapCluster(num2str(i)); 
      i
end

%%%%%%% LOAD BOOTSTRAP %%%%%%%%%%%%%%%%%%%

for b =1:B
    % load bootstrap
    load(['results/bootstrap/mdl' num2str(b) '.mat'])   
    % extract coefficients
    for t=1:4
        for m=1:3
            for i =1:16
                betaBA{t}(m,i,b)=betaB{t}(m,i);
            end
        end
    end
end

%%%%%%%%%%%%%%%%% COMPUTE BOOTSTRAP P VALUES %%%%%%%%%%%%%%%%%
for t=1:4
    for m=1:3
        for i =1:16
            bs=squeeze(betaBA{t}(m,i,:)); % create 1 array
            N1=length(find(bs<0)); % number coefficients under 0
            N2=length(find(bs>=0)); % number coefficients over 0
            pBootstrap{t}(m,i)=min(N1,N2)/B*2; % p value
            
        end
    end
end

% put in NaNs
for t=1:4
    pBootstrap{t}(2,1:2)=NaN;
    pBootstrap{t}(3,1:4)=NaN;
end


save('results/bootstrapResults.mat','pBootstrap');




%% CREATE R2 Tables and Figures

% standard error is a bit weird because few studies

%%%%% compare R2 between individual regressions, R2 mixed model and %%%%%%
%%%%  predicted R2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
clear all
clc

% load matlab functions
addpath(genpath('extraFiles'))

% load all data
load('results/singleFreqAnalysis.mat')
load('results/R2Pred.mat')
load('results/R2Pmeta.mat')

% put R2 together
R2Combined={R2Predicted,R2Seperate,R2Meta};
R2CombinedSE={R2PredictedSE,R2SeperateS,[]};

% places to save tables
filenameCorr = 'results/R2resultsCorr.xlsx';
filenameR2='results/R2results.xlsx';

% place to save figures
plotSave='plots/';

% all names and titles used in plots and tables
typeNames={'Psychological well-being','Depression','Borderline','Life satisfaction'};
namePlotTable={'Predicted_R2','Average_R2','MixedModel_R2'};
nameY={'Predicted \it{R}^2','Average \it{R}^2','multilevel model \it{{R}^2} '};
namesTypesR={'Regression results psychological well-being','Regression results depressive symptoms','Regression results life satisfaction'};
namesCorrelations={'Correlations all R2','Correlations R2 general well-being',...
    'Correlations R2 Depression','Correlations R2 Bordelrine','Correlations R2 Satisfaction with life'}; 
letters={'A.','B.','C.','D.'};

% variable for plots
rotateAngle=90; % rotation xlabel
alphaFigure=2.5; % size plot


% correlate for whole model and different type outcomes
R2All=[];
R2AllS=[];
idUse=find(isnan(R2Combined{1}{1}(:))==0);
for t=1:(size(dummyOutcome,2)+1)
    TempT=[R2Combined{1}{t}(idUse) R2Combined{2}{t}(idUse) R2Combined{3}{t}(idUse)];
    R2All=[R2All;TempT];    
    Ct{t}=corr(TempT); % correlate different type outcomes
end
C=corr(R2All); % correlate all possible R2

% write table of correlations
CTable=array2table(round(C,3));
CTable.Properties.VariableNames=namePlotTable;
CTable.Properties.RowNames=namePlotTable;
xlswrite(filenameCorr, cellstr(namesCorrelations{1}),1,['A' num2str(2)])
writetable(CTable,filenameCorr,'Sheet',1,'Range',['A' num2str(3)],'WriteRowNames',true);
for t=1:(size(dummyOutcome,2)+1)
    CTableTemp=array2table(round(Ct{t},3));
    CTableTemp.Properties.VariableNames=namePlotTable;
    CTableTemp.Properties.RowNames=namePlotTable;
    xlswrite(filenameCorr, cellstr(namesCorrelations{t+1}),1,['A' num2str(t*6+3-1)])
    writetable(CTableTemp,filenameCorr,'Sheet',1,'Range',['A' num2str(t*6+3)],'WriteRowNames',true);
end

%%%%%%%%%%%% PLOT R2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all

% loop over plots to make
for pi =1:length(R2Combined)
    figure('position',[0 0 500*alphaFigure 300*alphaFigure])
    co=get(gca,'colororder');
    % loop over types
    for t = 1:(size(dummyOutcome,2)+1)
        % plot
        subplot(2,2,t)
        dataPlot=R2Combined{pi}{t}';
        b2=bar(dataPlot);

        % plot standard errors
        if length(R2CombinedSE{pi})>0
            clear ctr
            clear ydt
            % extract x and y values of bar plots
            for k1 = 1:size(dataPlot,2)
                ctr(k1,:) = bsxfun(@plus, b2(1).XData, [b2(k1).XOffset]');
                ydt(k1,:) = b2(k1).YData;
            end 
            hold on
            % plot error bars
            errorBarMerijn(ctr, ydt, R2CombinedSE{pi}{t})
            hold off
        end

        % fix layout
        for i = 1:3
            b2(i).FaceColor = co(i,:)
        end
        set(gca,'xtick',[1:(length(names))],'xticklabel',names)
        if(t==1)
            legend({'Solo','Over M','Over M and SD'},'Location','northeast')
        end
        xtickangle(rotateAngle)
        ylabel(nameY{pi})
        title(namesTypesR{t})
        axis([1-1/3 size(R2Combined{pi}{1},2)+1/3 -0.08 0.3])
        set(gca, 'Ticklength', [0 0])
        grid off 
        
        % name panels a b c d
        text(-2,-0.1,letters{t},'FontSize',30)
    end
    
    % save figures
    saveas(gcf,[plotSave namePlotTable{pi} '.emf'])
    h=gcf;
    set(h,'PaperOrientation','landscape');
    print(gcf, '-dpdf', [plotSave namePlotTable{pi} '.pdf'])
end


%%%%%%%%% CREATE TABLES R2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lettersEx={'A','T','AM'}; % letters for excell columns
% loop over plots to make
for pi =1:length(R2Combined)
    % loop over type
    for t = 1:(size(dummyOutcome,2)+1)
        % create tables
        Ttemp=array2table(round(R2Combined{pi}{t},3));     
        Ttemp.Properties.VariableNames=namesX;
        Ttemp.Properties.RowNames={'solo','over_M','over_M_SD'};
        
        % write tables
        xlswrite(filenameR2, cellstr(namePlotTable{pi}),1,[lettersEx{pi} num2str((t-1)*(height(Ttemp)+4)+3-2)])
        xlswrite(filenameR2, cellstr(typeNames{t}),1,[lettersEx{pi} num2str((t-1)*(height(Ttemp)+4)+3-1)])
        writetable(Ttemp,filenameR2,'Sheet',1,'Range',[lettersEx{pi} num2str((t-1)*(height(Ttemp)+4)+3)],'WriteRowNames',true);
    end    
end



%% CREATE p Tables and Figures

close all
clear all
clc

load('results/singleFreqAnalysis.mat')
load('results/bootstrapResults.mat')
load('results/R2Pmeta.mat')


% put p-values together
pAll{1}=pMeta;
pAll{2}=pBootstrap;
pAll{3}=pMetaNoH;

% place to save figures
plotSave='plots/';

% place to save table
filenameP='results/pvalues.xlsx';

% all names and titles in plots and tables
typeNames={'Psychological well-being','Depression','Life satisfaction'};
namePlotTable={'meta_p','bootstrap_meta_p','weighted_meta_p'};
nameY={'p-values meta-analysis','bootstrap p-values meta-analysis','p-values weighted meta-analysis'};
namesTypesR={'Regression results psychological well-being','Regression results depressive symptoms',...
    'Regression results borderline symptoms','Regression results life satisfaction'};
letters={'A.','B.','C.','D.'};
comparisonName='compareNormalBootstrapWeighted';

% figure constants
cutOff=10^(-5);
plimit=0.05;
plimit2=0.001;
alphaFigure=2.5;
rotateAngle=90;


 %%%%%%%%%% PLOT P VALUES INDEPENDENTLY %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% create figure
close all
figure('position',[0 0 500*alphaFigure 300*alphaFigure])
co=get(gca,'colororder');

% loop over ways to compute p values
for pi =1:length(pAll)
    clf
    % loop over types
    for t=1:(size(dummyOutcome,2)+1)
        
        subplot(2,2,t) % create subplot
        
        % cut p-values at cutoff
        pMetaCut=pAll{pi}{t};
        pMetaCut(find(pMetaCut(:)<cutOff))=cutOff;

        % plot
        b=bar(pMetaCut');

        % plot lines of significant values
        hold on
        plot([2/3,size(pSolo,2)+1-2/3],[plimit,plimit],'color',co(4,:))
        plot([2/3,size(pSolo,2)+1-2/3],[plimit2,plimit2],'color',co(4,:))

        % do layout
        for i = 1:3
            b(i).FaceColor = co(i,:);
        end
        b(1).BaseValue = 1;
        set(gca,'YScale','log')
        set(gca,'xtick',[1:(length(names))],'xticklabel',names)
        set(gca,'ytick',[cutOff plimit2 0.05 1],'yticklabel',{['<' num2str(cutOff)],num2str(plimit2),'0.05','1'})
        xtickangle(rotateAngle)
        ylabel(nameY{pi});
        axis tight
        set(gca, 'Ticklength', [0 0])
        title(namesTypesR{t})
        if t==1
            legend({'Solo','Over M','Over M and SD'},'Location','southwest')
        end
        
        % name panels a b c d 
         text(-2,cutOff/4,letters{t},'FontSize',30) 
    end
    
    % save figures
    saveas(gcf,[plotSave namePlotTable{pi} '.emf'])
    h=gcf;
    set(h,'PaperOrientation','landscape');
    print(gcf, '-dpdf', [plotSave namePlotTable{pi} '.pdf'])      
end


%%%%%%%%%%%%%%%% PLOT COMPARISON OF P VALUES %%%%%%%%%%%%%%%%%%%%%%%%%

figure('position',[0 0 700*alphaFigure 150*alphaFigure])
% loop over ways to do p-values
for pi=1:3    
    subplot(1,3,pi)

    pMetaCut=pAll{pi}{1};
    pMetaCut(find(pMetaCut(:)<cutOff))=cutOff;
    
    % plot
    b=bar(pMetaCut');
    
    % plot lines
    hold on
    plot([2/3,size(pSolo,2)+1-2/3],[plimit,plimit],'color',co(4,:))
    plot([2/3,size(pSolo,2)+1-2/3],[plimit2,plimit2],'color',co(4,:))
    
    % layout
    for i = 1:3
        b(i).FaceColor = co(i,:);
    end
    b(1).BaseValue = 1;
    set(gca,'YScale','log');
    set(gca,'xtick',[1:(length(names))],'xticklabel',names);
    set(gca,'ytick',[cutOff plimit2 0.05 1],'yticklabel',{['<' num2str(cutOff)],num2str(plimit2),'0.05','1'});
    xtickangle(rotateAngle);
    ylabel(nameY{pi});
    axis tight
    set(gca, 'Ticklength', [0 0]);
    
    if t==1
        legend({'Solo','Over M','Over M and SD'},'Location','southwest')
    end
    
    % name panels a b c d 
    text(-2,cutOff/4,letters{pi},'FontSize',30) ;
end

saveas(gcf,[plotSave comparisonName '.emf']);
h=gcf;
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(h,[plotSave comparisonName '.pdf'],'-dpdf','-r0');


%%%%%%%%% CREATE TABLES p %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lettersEx={'A','T','AM'}; % letters for excell columns
% loop over plots to make
for pi =1:length(pAll)
    % loop over type
    for t = 1:(size(dummyOutcome,2)+1)
        % create tables
        Ttemp=array2table(pAll{pi}{t});     
        Ttemp.Properties.VariableNames=namesX;
        Ttemp.Properties.RowNames={'solo','over_M','over_M_SD'};
        
        % write tables
        xlswrite(filenameP, cellstr(namePlotTable{pi}),1,[lettersEx{pi} num2str((t-1)*(height(Ttemp)+4)+3-2)]);
        xlswrite(filenameP, cellstr(typeNames{t}),1,[lettersEx{pi} num2str((t-1)*(height(Ttemp)+4)+3-1)]);
        writetable(Ttemp,filenameP,'Sheet',1,'Range',[lettersEx{pi} num2str((t-1)*(height(Ttemp)+4)+3)],'WriteRowNames',true);
    end    
end


%% FREQUENTIST ANALYSIS CLASSIFICATION OUTCOME

% no classification outcomes in online package

% close all
% clear all
% clc
% 
% % load data
% load('results/initAll.mat')
% 
% 
% %%%%%%%%%%%%%% COMPUTE LOGISTIC REGRESSION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% add=1;
% % loop over data sets
% for iDataset=1:length(dataStrAll)
%     % loop over classification outcomes
%     if length(CAll{iDataset})>0
%         for j=1:(size(CAll{iDataset},2)-1)
%             % find right statistics for right participants
%             idNoNan=find(isnan(CAll{iDataset}(:,j+1))==0);
%             Y=CAll{iDataset}(idNoNan,j+1);
%             Stemp=SAll{iDataset}(idNoNan,:);
%             
%             % standardize
%             S=zeros(size(Stemp,1),size(Stemp,2));
%             for k =1:size(Stemp,2)
%                 S(:,k)=(Stemp(:,k)-nanmean(Stemp(:,k)))/nanstd(Stemp(:,k));
%             end            
%           
%             % test if enough participants in each group AND don't do SCID_MDD_BPD
%             LengthParts=[length(find(Y==0)), length(find(Y==1))]/length(Y);           
%             if min(LengthParts)>0.05 & strcmp(Cnames{iDataset,j},'SCID_MDD_BPD')==0 
%                 
%                 % mean and sd statistics
%                 SetM=[1 2]; % MEAN COLUMNS
%                 SetSD=[1 2 3 4]; % MEAN AND SD COLUMNS
%                 SM=[S(:,SetM)];
%                 SSD=[S(:,SetSD)];
% 
%                 % fit model with mean
%                 mdlM=fitglm(SM,Y,'Distribution','binomial','Link','logit');
%                 scores = mdlM.Fitted.Probability;
%                 [XA,YA,T,AUC] = perfcurve(Y,scores,1);
%                 R2MC(add)=AUC;
% 
%                 % fit model with mean and sd
%                 mdlSD=fitglm(SSD,Y,'Distribution','binomial','Link','logit');
%                 scores = mdlSD.Fitted.Probability;
%                 [XA,YA,T,AUC] = perfcurve(Y,scores,1);
%                 R2SDC(add)=AUC;
% 
%                 % fit solo models
%                 for k=1:size(S,2)
%                     SSolo=[S(:,k)];
%                     mdlSolo=fitglm(SSolo,Y,'Distribution','binomial','Link','logit');       
%                     scores = mdlSolo.Fitted.Probability;
%                     [XA,YA,T,AUC] = perfcurve(Y,scores,1);
%                     R2SoloC(add,k)=AUC;
%                     pSoloC(add,k)=mdlSolo.Coefficients.pValue(end);
% 
%                     BetaSoloC(add,k)=mdlSolo.Coefficients.Estimate(end);
%                     SESoloC(add,k)=mdlSolo.Coefficients.SE(end);
%                 end
% 
%                 % fit over mean models
%                 for k=3:size(S,2)
%                     SOverM=[SM S(:,k)];
%                     mdlOverM=fitglm(SOverM,Y,'Distribution','binomial','Link','logit');
%                     scores = mdlOverM.Fitted.Probability;
%                     [XA,YA,T,AUC] = perfcurve(Y,scores,1);
%                     R2WithMC(add,k-2)=AUC;
%                     R2OverMC(add,k-2)=R2WithMC(add,k-2)-R2MC(add);
%                     pOverMC(add,k-2)=mdlOverM.Coefficients.pValue(end);
% 
%                     BetaOverMC(add,k)=mdlOverM.Coefficients.Estimate(end);
%                     SESOverMC(add,k)=mdlOverM.Coefficients.SE(end);
%                 end
% 
%                 % fit over m and sd models
%                 for k=5:size(S,2)
%                     SOverMSD=[SSD S(:,k)];
%                     mdlOverMSD=fitglm(SOverMSD,Y,'Distribution','binomial','Link','logit');
%                     scores = mdlOverMSD.Fitted.Probability;
%                     [XA,YA,T,AUC] = perfcurve(Y,scores,1);
%                     R2WithMSDC(add,k-4)=AUC;
%                     R2OverMSDC(add,k-4)=R2WithMSDC(add,k-4)-R2SDC(add);
%                     pOverMSDC(add,k-4)=mdlOverMSD.Coefficients.pValue(end);
% 
%                     BetaOverMSDC(add,k)=mdlOverMSD.Coefficients.Estimate(end);
%                     SESOverMSDC(add,k)=mdlOverMSD.Coefficients.SE(end);
%                 end
%                 
%             YMixedCell{add}=Y;
%             SMixedCell{add}=S;
%             GDatasetCell{add}=ones(length(Y),1)*iDataset;
% 
%             % give name to rows
%             nameRowC{add}=[dataStrPaper{iDataset} ': ' Cnames{iDataset,j}]
%             NNC(add,1)=size(S,1); % number of participants  
%             add=add+1;
%             end
%         end
%     end    
% end
% 
% 
% SMixedC=[];
% YMixedC=[];
% GMixedC=[];
% GDatasetC=[];
% for i = 1:length(SMixedCell)
%     SMixedC=[SMixedC;SMixedCell{i}];
%     YMixedC=[YMixedC;YMixedCell{i}];
%     GMixedC=[GMixedC;ones(length(YMixedCell{i}),1)*i];
%     GDatasetC=[GDatasetC;GDatasetCell{i}];
% end
% 
% R2Seperate=NaN(3,size(SMixedC,2));
% 
% R2Seperate(1,:)=R2SoloC'*NNC/sum(NNC);
% R2Seperate(2,3:end)=R2OverMC'*NNC/sum(NNC);
% R2Seperate(3,5:end)=R2OverMSDC'*NNC/sum(NNC);
% 
% save('results/singleFreqAnalysisC.mat')


%% Generalized Linear Mixed Model Analysis (multilevel logistic regression) for classiciation

% no classification outcomes in online package
% 
% close all
% clear all
% clc
% 
% % load data
% load('results/singleFreqAnalysisC.mat')
% 
% 
% % path where to save assumption test plots
% basicPath='C:/Users/u0084978/SyncFiles/Personal/werk/beat the mean/Major Revision/paper';
% 
% % create empty arrays
% pMetaC=NaN(3,size(SMixedC,2));
% R2MetaC=NaN(3,size(SMixedC,2));
% R2Predict=NaN(3,size(SMixedC,2));
% R2PredictSE=NaN(3,size(SMixedC,2));
% 
% % create everything for meta resuls appendix
% namesLatex=names;
% namesLatex{11}=['\' namesLatex{11}];
% stringLatexAssumptions=''; % empty string to fill with appendix text
% stringLatexTextModel='\\section{Classification}';
% stringLatexTableBegin=[char(10) '\\begin{table}[H]' char(10) ...
% '\\begin{center}' char(10) ...
% '\\begin{tabular}{l r r r r r r r}' char(10) ...
% '& estimate & SE & tstat & DF & pval & 95\\%% conf int lower bound  & 95\\%% conf int upper bound \\\\' char(10) ...
% '\\hline' char(10)];
% stringLatexTableEnd='\\end{tabular} \\end{center} \\end{table} \\clearpage';
% 
% 
% % loop over models (solo, over M and over MSD)
% for m=1:3   
%     
%     % basis: over what do we predict
%     % create Formula String for basis and extra
%     % Table looks like "Y|X1|X2|...|Xn|G"
%     % first column (Var1) is outcome variable, next columns are predictors
%     % and last column is group variable
%     if m==1
%         basis=[];
%         Xbes= 'Var1~1+Var2+(1+Var2|Var3)';
%         stringLatexTextModel=[ stringLatexTextModel '\\subsection{Solo variables}' char(10)];
%     elseif m==2
%         basis=[1 2];
%         Xbs= 'Var1~1+Var2+Var3+(1+Var2+Var3|Var4)';
%         Xbes= 'Var1~1+Var2+Var3+Var4+(1+Var2+Var3+Var4|Var5)';
%         stringLatexTextModel=[stringLatexTextModel '\\subsection{Over M}' char(10)];
%     else
%         basis=[1 2 3 4];
%         Xbs= 'Var1~1+Var2+Var3+Var4+Var5+(1+Var2+Var3+Var4+Var5|Var6)';
%         Xbes= 'Var1~1+Var2+Var3+Var4+Var5+Var6+(1+Var2+Var3+Var4+Var5+Var6|Var7)';
%         stringLatexTextModel=[stringLatexTextModel '\\subsection{Over M and SD}' char(10)];
%     end    
%     
%     % create new table and empty latex file
%     tableLatex=['\\subsubsection{Overview ' 'psychological well-being' '}' char(10) stringLatexTableBegin];
%     modelLatexTemp='';
%         
% 
% %     create the basis
%     if length(basis)>0
%         X=[SMixedC(:,basis)];
%         T=array2table([YMixedC X GMixedC]); % create table of all variables
%         mdl=fitglme(T,Xbs,'Distribution','Binomial'); % fit model
%         R2Basis=1-mean(abs(YMixedC-round(mdl.fitted))); % percentage correct
%         errorBasisPredict=R2GenMixedCV(T,Xbs); % predicted errors
%         R2BasisPredict=1-mean(errorBasisPredict); % predicted percentage correct
%     end
% 
%     % go over all dynamics not in basis
%     for j =(length(basis)+1):size(SMixedC,2)
% 
%         % create data for model
%         X=[SMixedC(:,basis) SMixedC(:,j)];
%         T=array2table([YMixedC X GMixedC]); % create table of all variables
%         
%         % create names for model and forumla
%         clear namesTable
%         namesTable={'Y'};
%         for bi=1:length(basis)
%             namesTable{bi+1}=namesX{bi};
%         end
%         namesTable{length(basis)+2}=namesX{j};
%         namesTable{length(basis)+3}='Questionnaire';
%         T.Properties.VariableNames=namesTable;      
%         XbesTemp=Xbes;
%         for bi=1:size(T,2)
%             sTemp=['Var' num2str(bi)];
%             XbesTemp = strrep(XbesTemp,sTemp,namesTable{bi});
%         end
%         
%         % estimate model
%         mdl=fitglme(T,XbesTemp,'Distribution','Binomial'); % fit model
%         
%         % put everything in latex table
%         tableLatex=[tableLatex '$' namesLatex{j} '$' '&' num2str(mdl.Coefficients.Estimate(end),4) '&' num2str(mdl.Coefficients.SE(end),4) '&'...
%             num2str(mdl.Coefficients.tStat(end),4) '&' num2str(mdl.Coefficients.DF(end),4) '&' num2str(mdl.Coefficients.pValue(end),4)...
%             '&' num2str(mdl.Coefficients.Lower(end),4) '&' num2str(mdl.Coefficients.Upper(end),4)  '\\\\' char(10)];
%         modelLatexTemp=[modelLatexTemp '\\subsubsection{$' namesLatex{j} '$: general psychological well-being}' char(10)];
%         stringModelTemp=evalc('disp(mdl)');
%         stringModelTemp2 = strrep(stringModelTemp,'%','%%');
%         stringModelTemp3 = strrep(stringModelTemp2,'<strong>','');
%         stringModelTemp4 = strrep(stringModelTemp3,'</strong>','');
%         modelLatexTemp=[modelLatexTemp '\\begin{lstlisting}'  stringModelTemp4 char(10) '\n \\end{lstlisting} \\clearpage'];
% 
%         % extract p value      
%         pMetaC(m,j)=mdl.Coefficients.pValue(end);   % p value statistic
%  
%         % compute effect sizes
%         errors=abs(YMixedC-round(mdl.fitted)); %  extract errors
%         errorPredict=R2GenMixedCV(T,XbesTemp); % extract predicted errors
%         
%         R2MixedTemp=1-mean(errors); % percentage correct
%         R2MixedTempPredict=1-mean(errorPredict); % predicted percentage correct
%         
%         if length(basis>0)
%             R2MetaC(m,j)=R2MixedTemp-R2Basis; % added percentage correct
%             R2Predict(m,j)=R2MixedTempPredict-R2BasisPredict; % added percentage predicted correct
%             R2PredictSE(m,j)=WithinBetweenSE((errorBasisPredict-errorPredict),GMixedC); % standard error
%         else
%             R2MetaC(m,j)=R2MixedTemp; % percentage correct
%             R2Predict(m,j)=R2MixedTempPredict; % predicted percentage correct
%             R2PredictSE(m,j)=WithinBetweenSE(errorPredict,GMixedC); % standard error of predicted percentage
%         end        
%         
%         
%         disp([num2str(((m-1)*size(SMixedC,2)+j)/(3*size(SMixedC,2))*100) '% Completed'])
%     end
%     
%     % put latex files together
%     stringLatexTextModel=[stringLatexTextModel char(10) tableLatex stringLatexTableEnd];
%     stringLatexTextModel=[stringLatexTextModel char(10) modelLatexTemp char(10)];
%     
% end
% 
% % save latex file
% fid = fopen([basicPath '/rawModelC.tex'],'wt');
% fprintf(fid, stringLatexTextModel);
% fclose(fid);
% 
% 
% save('results/R2PmetaC.mat','R2MetaC','pMetaC','R2Predict','R2PredictSE');

 %% CREATE R2 Tables and Figures Classification
 
 % no classification outcomes in online package
% 
% close all
% clear all
% clc
% 
% % load all data
% load('results/R2PmetaC.mat')
% load('results/singleFreqAnalysisC.mat')
% 
% % put R2 together
% R2Combined={R2Predict,R2Seperate,R2MetaC};
% R2CombinedSE={R2PredictSE,[],[]};
% 
% % put p-values together
% pAll{1}=pMetaC;
% pAll{2}=pMetaC;
% pAll{3}=pMetaC;
% 
% 
% % places to save tables
% filenameCorr = 'results/R2resultsCorrClass.xlsx';
% filenameR2='results/R2resultsClass.xlsx';
% filenameP='results/pvaluesClass.xlsx';
% 
% % place to save figures
% plotSave='plots/';
% 
% % all names and titles used in plots and tables
% namePlotTable={'Predicted_Accuracy_Class','Average_AUC_Class','MixedModel_Fitted_Accuracy_Class'};
% nameY={'Predictive accuracy','Average AUC','Fitted accuracy'};
% namesCorrelations={'Correlations all R2'}; 
% letters={'A.','B.','C.','D.'};
% 
% 
% % variable for plots
% rotateAngle=90; % rotation xlabel
% alphaFigure=2.5; % size plot
% 
% % compute correlations between R2
% idUse=find(isnan(R2Combined{1}(:))==0);
% R2All=[];
% for i =1:length(R2Combined)
%     R2All=[R2All R2Combined{i}(idUse)];
% end
% C=corr(R2All)
% 
% CTable=array2table(round(C,3));
% CTable.Properties.VariableNames=namePlotTable;
% CTable.Properties.RowNames=namePlotTable;
% xlswrite(filenameCorr, cellstr(namesCorrelations{1}),1,['A' num2str(2)])
% writetable(CTable,filenameCorr,'Sheet',1,'Range',['A' num2str(3)],'WriteRowNames',true);
% %%%%%%%%%%%% PLOT R2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% close all
% 
% 
% % loop over plots to make
% for pi =1:length(R2Combined)
%     fig{pi}=figure('position',[0 0 400*alphaFigure 190*alphaFigure])
%     co=get(gca,'colororder');
%     % loop over types
%         subplot(1,2,1)
%         dataPlot=R2Combined{pi}';
%         b2=bar(dataPlot);
% 
%         % plot standard errors
%         if length(R2CombinedSE{pi})>0
%             clear ctr
%             clear ydt
%             % extract x and y values of bar plots
%             for k1 = 1:size(dataPlot,2)
%                 ctr(k1,:) = bsxfun(@plus, b2(1).XData, [b2(k1).XOffset]');
%                 ydt(k1,:) = b2(k1).YData;
%             end 
%             hold on
%             % plot error bars
%             errorBarMerijn(ctr, ydt, R2CombinedSE{pi})
%             hold off
%         end
% 
%         % fix layout
%         for i = 1:3
%             b2(i).FaceColor = co(i,:)
%         end
%         set(gca,'xtick',[1:(length(names))],'xticklabel',names)
% 
%         legend({'Solo','Over M','Over M and SD'},'Location','northeast')
%         xtickangle(rotateAngle)
%         ylabel(nameY{pi})
%         title('Effect size classification psychological wellbeing')
%         x0=1-1/3;
%         xe=size(R2Combined{pi},2)+1/3;
%         y0=-0.05;
%         ye=1;       
%         axis([x0 xe y0 ye])
%         set(gca, 'Ticklength', [0 0])
%         grid off 
%         
%         % name panels a b c d
% %         text(x0-(xe-x0)/5*0.8,y0-(ye-y0)/2.5,letters{1},'FontSize',30);
%         text(x0-(xe-x0)/10,y0-(ye-y0)/10,letters{1},'FontSize',30);
% end
% 
% 
% %%%%%%%%% CREATE TABLES R2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lettersEx={'A','T','AM'}; % letters for excell columns
% % loop over plots to make
% for pi =1:length(R2Combined)
%         % create tables
%         Ttemp=array2table(round(R2Combined{pi},3));     
%         Ttemp.Properties.VariableNames=namesX;
%         Ttemp.Properties.RowNames={'solo','over_M','over_M_SD'};
%         
%         % write tables
%         xlswrite(filenameR2, cellstr(namePlotTable{pi}),1,[lettersEx{pi} num2str(2)])
%         writetable(Ttemp,filenameR2,'Sheet',1,'Range',[lettersEx{pi} num2str(4)],'WriteRowNames',true);
% end
%     
% 
% 
% %%%%%%%%%%%%% PLOT P %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % figure constants
% cutOff=10^(-5);
% plimit=0.05;
% plimit2=0.001;
% 
% % loop over ways to compute p values
% for pi =1:length(pAll)
%     figure(fig{pi})
%     subplot(1,2,2) % create subplot
%         
%     % cut p-values at cutoff
%     pMetaCut=pAll{pi};
%     pMetaCut(find(pMetaCut(:)<cutOff))=cutOff;
% 
%     % plot
%     b=bar(pMetaCut');
% 
%     % plot lines of significant values
%     hold on
%     plot([2/3,size(pAll{pi},2)+1-2/3],[plimit,plimit],'color',co(4,:))
%     plot([2/3,size(pAll{pi},2)+1-2/3],[plimit2,plimit2],'color',co(4,:))
% 
%     % do layout
%     for i = 1:3
%         b(i).FaceColor = co(i,:);
%     end
%     b(1).BaseValue = 1;
%     set(gca,'YScale','log')
%     set(gca,'xtick',[1:(length(names))],'xticklabel',names)
%     set(gca,'ytick',[cutOff plimit2 0.05 1],'yticklabel',{['<' num2str(cutOff)],num2str(plimit2),'0.05','1'})
%     xtickangle(rotateAngle)
%     ylabel('p-value');
%     
%     x0=1-1/3;
%     xe=size(R2Combined{pi},2)+1/3;
%     y0=cutOff;
%     ye=1;     
%     axis([x0 xe y0 ye])
%     set(gca, 'Ticklength', [0 0])
%     title('p-value classification psychological wellbeing')
% 
%     % name panels a b c d 
%      text(x0-(xe-x0)/9,y0/3.1,letters{2},'FontSize',30);
%      
%     % save figures
%     saveas(gcf,[plotSave namePlotTable{pi} '.emf'])
%     h=gcf;
%     set(h,'Units','Inches');
%     pos = get(h,'Position');
%     set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
%     print(h,[plotSave namePlotTable{pi} '.pdf'],'-dpdf','-r0');  
% end
% 
% 
% 
% %%%%%%%%% CREATE TABLES p %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % loop over plots to make
% 
% Ttemp=array2table(pAll{pi});     
% Ttemp.Properties.VariableNames=namesX;
% Ttemp.Properties.RowNames={'solo','over_M','over_M_SD'};
% writetable(Ttemp,filenameP,'Sheet',1,'Range','A1','WriteRowNames',true);
