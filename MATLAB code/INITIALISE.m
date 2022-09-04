%% READ DATA AND COMPUTE ALL STATS
close all
clear all
clc

%%%%%%%%%%%%%%%%%%%%%%%   DECLARATION PARAMTERS    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% all data sets for this package
dataStrAll={'PETEMADDY'};
        
% rename data sets for output             
    for st=1:length(dataStrAll)
        dataTemp=dataStrAll{st};
        switch dataTemp
            case 'MDD GOTLIB'
                dataStrPaper{st}='Thompson 2012';
            case 'KATHLEEN STRESSKLINIEK'
                dataStrPaper{st}='Van der Gucht 2017';
            case 'MTURK DAILY DIARY'
                dataStrPaper{st}='Dejonckheere 2017';
            case 'Cogito'
                dataStrPaper{st}='Schmiedek 2010';
            case 'Laura ESM 2016'
                dataStrPaper{st}='Sels 2018';
            case 'Laura ESM 2014'
                dataStrPaper{st}='Sels 2017';
            case 'MARLIES BPD'
                dataStrPaper{st}='Houben 2016';
            case 'PETEMADDY'
                dataStrPaper{st}='Koval 2013';
            case 'CLINICAL ESM'
                dataStrPaper{st}='Dejonckheere in prep';
            case 'ELISE ESM14'
                dataStrPaper{st}='Kalokerinos in prep';
            case 'LONGITUDINAL W1'
                dataStrPaper{st}='Pe 2016 Wave 1';
            case 'LONGITUDINAL W2'
                dataStrPaper{st}='Pe 2016 Wave 2';
            case 'LONGITUDINAL W3'
                dataStrPaper{st}='Pe 2016 Wave 3';
            case 'MDD BPD TRULL'
                dataStrPaper{st}='Trull 2008';
            case 'JULIAN EGON'
                dataStrPaper{st}='Provenzano in prep';
            otherwise
                warning('not encoded');
        end
    end        
        

% all measures to compute
statisticsString={'MEAN','STD','RELSTD','RMSSD','AR','CORRPANA','ICC','DENSITY','GINI'};

minCompliance=0.5;  % minimum compliance for participants
boundsAll=[0 100];  % transform all variables to these bounds
GINILimit=10;  % limit of present of emotion. > GINILimit means emotion is active
minL=20;  % minimum length time series part that can be used for AR and MSSD purposes.       

%addpath(genpath('extraFiles')) # i included them via pathtool
%addpath(genpath('statistics')) # i included them via pathtool

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   BEGIN COMPUTATION DYNAMICAL MEASURES  %%%%%%%%%%%%%%%%        
addPerson=1; % counter for all participants

% Loop over all data sets
for iDataset=1:length(dataStrAll)
    
    % current data file name base
    dataStr=dataStrAll{iDataset};

    % read min and max of emotion measurements
    file=['C:\Users\sotir\Desktop\Thesis_papers\Emotion Recognition in Conversations\Copmlex affect dynamics\Reproducible MATLAB Code\data\' dataStr ' bound.csv'];
    imp=importdata(file,';',1);
    bounds=imp.data;

    % read data regression outcome
    file=['C:\Users\sotir\Desktop\Thesis_papers\Emotion Recognition in Conversations\Copmlex affect dynamics\Reproducible MATLAB Code\data\' dataStr ' reg.csv'];
    if exist(file)
        % regression outcome data
        imp=importdata(file,';',1);
        Rfull=imp.data;
        rTrue=1;

        % regression outcome variable names
        fid = fopen(file,'r');
        headerFile=fgetl(fid);
        headerFile=headerFile(4:end);
        nameR = strsplit(headerFile,';');
        fclose(fid);   
        for j=2:length(nameR)
            Rnames{iDataset,j-1}=nameR{j};
        end 
    else
        Rnames{iDataset,1}='';
        Rfull=[];
    end

    % read data classification outcomes
    file=['C:\Users\sotir\Desktop\Thesis_papers\Emotion Recognition in Conversations\Copmlex affect dynamics\Reproducible MATLAB Code\data\' dataStr ' class.csv'];
    if exist(file)
        % classification data
        imp=importdata(file,';',1);
        Cfull=imp.data;
        cTrue=1;

        % classification outcome variable names
        fid = fopen(file,'r');
        headerFile=fgetl(fid);
        headerFile=headerFile(4:end);
        nameC = strsplit(headerFile,';');
        fclose(fid); 
        for j=2:length(nameC)
            Cnames{iDataset,j-1}=nameC{j};
        end
    else
        Cnames{iDataset,1}='';
        Cfull=[];       
    end

    % read Valence of Emotions
    file=['C:\Users\sotir\Desktop\Thesis_papers\Emotion Recognition in Conversations\Copmlex affect dynamics\Reproducible MATLAB Code\data\' dataStr ' val.csv'];
    imp=importdata(file,';',1);
    Val=imp.data;

    % read data file with PA, NA and individual emotions
    file=['C:\Users\sotir\Desktop\Thesis_papers\Emotion Recognition in Conversations\Copmlex affect dynamics\Reproducible MATLAB Code\data\' dataStr '.csv'];
    imp=importdata(file,';',1);
    X=imp.data;
    subj=X(:,1); % subject column
    unit=X(:,2); % unit column
    beepno=X(:,3); % beepno column
   
    
    % read names emotions
    fid = fopen(file,'r');
    headerFile=fgetl(fid);
    headerFile=headerFile(4:end);
    namesEmotionsTemp = strsplit(headerFile,';');
    fclose(fid);    
    clear namesEmotions
    for i=6:length(namesEmotionsTemp)
        namesEmotions{i-5}=namesEmotionsTemp{i};
    end
    
    % read Covariances
    file=['C:\Users\sotir\Desktop\Thesis_papers\Emotion Recognition in Conversations\Copmlex affect dynamics\Reproducible MATLAB Code\data\' dataStr ' cov.csv'];
    imp=importdata(file,';',1);
    COVDataset(iDataset,:)=imp.data(4:end); % not interested in first 3 covariates
    
    fid = fopen(file,'r');
    headerFile=fgetl(fid);
    headerFile=headerFile(4:end);
    namesCovTemp = strsplit(headerFile,';');
    for i =4:length(namesCovTemp) % not interested in first 3 covariates
        namesCov{i-3}=namesCovTemp{i};
    end
    fclose(fid); 
    
   
    % transform the data to chosen bounds
    PANA=(X(:,4:5)-bounds(1))/(bounds(2)-bounds(1))*(boundsAll(2)-boundsAll(1))+boundsAll(1);
    E=(X(:,6:end)-bounds(1))/(bounds(2)-bounds(1))*(boundsAll(2)-boundsAll(1))+boundsAll(1);
        
    
 
    % preprocessing: remove subjects with not enough data points
    % AND do some basic checks on data
    [subjNotUse subj unit beepno PANA E Rfull Cfull deletedInfo dataPointsInfo]=preprocessData(subj,unit,beepno,PANA,E,boundsAll,...
        Rfull,Cfull,minL,Val,namesEmotions,minCompliance,GINILimit,COVDataset(end,:));
    
    % inter vs between variance PA and NA        
    [infoIcc]=iccPana(subj,PANA);    
    ui=unique(subj);

    % write idx and long format for spss check
    %writeDataLongFormat( subj,E,Val,dataStr,beepno)

    % define dynamic measures
    strPANA={'PA','NA'};
    addst=1;
    for st=1:length(statisticsString)
        statTemp=statisticsString{st};
        switch statTemp
            case 'MEAN'
                statistics{addst}=@()MEAN(PANA,subj,strPANA);
            case 'STD'
                statistics{addst}=@()STD(PANA,subj,strPANA);
            case 'RELSTD'
                statistics{addst}=@()RELSTD(PANA,subj,boundsAll(1),boundsAll(2),strPANA);
            case 'RMSSD'
                statistics{addst}=@()RMSSD(PANA,subj,unit,beepno,strPANA);
            case 'AR'
                statistics{addst}=@()AR(PANA,subj,unit,beepno,strPANA);
            case 'CORRPANA'
                statistics{addst}=@()CORRPANA(PANA,subj);
            case 'ICC'
                statistics{addst}=@()ICC(E,subj,Val,strPANA);
            case 'DENSITY'
                statistics{addst}=@()DENSITY( E,subj,unit,beepno,dataStr);
            case 'GINI'
                statistics{addst}=@()GINI(E,subj,Val,strPANA,GINILimit);
            otherwise
                error(['Measure ' statTemp ' unknown']);
        end
        addst=addst+1;
    end

    % compute measures
    S=[];
    names={};
    add=1;
    Lworst=ones(length(unique(subj)),1)*inf;
    for i =1:length(statistics)
        [temp namesTemp L]=statistics{i}();
        Lworst=min(Lworst,min(L,[],2));
        S=[S temp];
        for j =1:length(namesTemp)
            names{add}=namesTemp{j};
            add=add+1;
        end       
    end
    
    % save time series
    for i =1:length(ui)
        idx=find(subj==ui(i));
        for v=-1:2:1
            idv=find(Val==v);
            personTimeSeries{addPerson,v/2+1.5}=E(idx,idv);
        end
        addPerson=addPerson+1;
    end
    
    % extra information data set
    [ ~,~,~,nNegativeICC ] = ICC( E,subj,Val,strPANA );
    [ ~,~,~,nPositiveRho ] = CORRPANA(PANA,subj);
    infoDataset(iDataset,:)=[dataPointsInfo deletedInfo infoIcc nNegativeICC,nPositiveRho];
    
    % keep dependent and independent variables
    RAll{iDataset}=Rfull;
    CAll{iDataset}=Cfull;
    SAll{iDataset}=S;
    
    disp([num2str(iDataset/length(dataStrAll)*100) '% Completed'])
end

% redo names for matlab purposses
for i = 1:length(names)
    namesX{i}=strrep(names{i},'{','');
    namesX{i}=strrep(namesX{i},'}','');
    namesX{i}=strrep(namesX{i},'\','');
    namesX{i}=strrep(namesX{i},'^*','r');
end

% check for NAN
for i =1:length(dataStrAll)
    LN(i)=length(find(isnan(SAll{i}(:))==1));
end
if max(LN)>0
    error('NaNs found in statistics')
end

% save data
%save('results/initAll.mat','SAll','dataStrPaper','names','minL','statistics','Cnames','Rnames','CAll','RAll','dataStrAll','namesX','COVDataset','namesCov')
%save('results/allSeries.mat','personTimeSeries');

% output in tables
T=array2table(infoDataset);
T.Properties.VariableNames={'meanCompliance','meanBeepM','totalSubj','subjKept','subjDeleted',...
    'lowCompliance','notEnoughData','noVariance','noPosOrNegEmoFreq','ICCPA','ICCNA','negativeICC','positiveRho'};
T.Properties.RowNames=dataStrAll;

filename = 'results/dataInfo.xlsx';
%writetable(T,filename,'Sheet',1,'Range',['A1'],'WriteRowNames',true)

