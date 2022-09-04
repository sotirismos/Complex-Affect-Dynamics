%% READ DATA AND COMPUTE ALL STATS
close all
clear 
clc

%%%%% DECLARATION PARAMETERS  %%%%%
% all data sets for this package
dataStrAll={'KEMOCON'};
        
% rename data sets for output             
    for st=1:length(dataStrAll)
        dataTemp=dataStrAll{st};
        switch dataTemp
            case 'KEMOCON'
                dataStrPaper{st}='Hadjileontiadis et al. 2020'; 
            otherwise
                warning('not encoded');
        end
    end        
        
% all measures to compute
statisticsString={'MEAN','STD','RELSTD','RMSSD','AR'};

boundsAll=[1 100];  % transform all variables to these bounds
minL=20;  % minimum length time series part that can be used for AR and MSSD purposes.       

%%%%% BEGIN COMPUTATION DYNAMICAL MEASURES %%%%%        
addPerson=1; % counter for all participants

% Loop over all data sets
for iDataset=1:length(dataStrAll)
    
    % current data file name base
    dataStr=dataStrAll{iDataset};

    % read min and max of emotion measurements
    file=['C:\Users\sotir\Desktop\Thesis_papers\Emotion Recognition in Conversations\Copmlex affect dynamics\Reproducible MATLAB Code\data\KEmoCon\partner\' dataStr ' bound.csv'];
    imp=importdata(file,';',1);
    bounds=imp.data;

    % read Valence of Emotions
    file=['C:\Users\sotir\Desktop\Thesis_papers\Emotion Recognition in Conversations\Copmlex affect dynamics\Reproducible MATLAB Code\data\KEmoCon\partner\' dataStr ' val.csv'];
    imp=importdata(file,';',1);
    Val=imp.data;

    % read data file with PA, NA and individual emotions
    file=['C:\Users\sotir\Desktop\Thesis_papers\Emotion Recognition in Conversations\Copmlex affect dynamics\Reproducible MATLAB Code\data\KEmoCon\partner\' dataStr '.csv'];
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
   
    % transform the data to chosen bounds
    PANA=(X(:,4:5)-bounds(1))/(bounds(2)-bounds(1))*(boundsAll(2)-boundsAll(1))+boundsAll(1);
    E=(X(:,6:end)-bounds(1))/(bounds(2)-bounds(1))*(boundsAll(2)-boundsAll(1))+boundsAll(1);

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
                statistics{addst}=@()DENSITY(E,subj,unit,beepno,dataStr);
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
        [temp, namesTemp, L]=statistics{i}();
        Lworst=min(Lworst,min(L,[],2));
        S=[S temp];
        for j =1:length(namesTemp)
            names{add}=namesTemp{j};
            add=add+1;
        end       
    end
    
    ui = unique(subj);
    % save time series
    for i =1:length(ui)
        idx=find(subj==ui(i));
        for v=-1:2:1
            idv=find(Val==v);
            personTimeSeries{addPerson,v/2+1.5}=E(idx,idv);
        end
        addPerson=addPerson+1;
    end
    
    % keep variables
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

% plot PA time series
%x = linspace(0,100); % time steps
%x1 = PANA(1:100,1);
%x2 = PANA(171:270,1);
%x3 = PANA(341:440,1);
%x4 = PANA(477:576,1);
%figure;
%plot(x,x1,x,x2,x,x3,x,x4);
%title('PA time series for 100 5s time steps for participants 1-4');
%xlabel('Time steps');
%ylabel('PA');

% plot NA time series
%x = linspace(0,100); % time steps
%x1 = PANA(1:100,2);
%x2 = PANA(171:270,2);
%x3 = PANA(341:440,2);
%x4 = PANA(477:576,2);
%figure;
%plot(x,x1,x,x2,x,x3,x,x4);
%title('NA time series for 100 5s time steps for participants 1-4');
%xlabel('Time steps');
%ylabel('NA'); 

% plot distribution of complex affect dynamics
%nbins = 20;
%histogram(S(:,10), nbins)
%xlabel('AR NA')
%ylabel('Frequency')
%title('Distribution of AR NA-partner perspective')

% check for NAN
for i =1:length(dataStrAll)
    LN(i)=length(find(isnan(SAll{i}(:))==1));
end
if max(LN)>0
    error('NaNs found in statistics')
end

% save data
%save('C:\Users\sotir\Documents\thesis\affect_dynamics\results\external\initAll.mat','SAll','dataStrPaper','names','minL','statistics','dataStrAll','namesX')
%save('C:\Users\sotir\Documents\thesis\affect_dynamics\results\external\allSeries.mat','personTimeSeries');
