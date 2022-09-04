function [ mv ] = maximumVAR( M,MIN,MAX,n)
% 
% Description
% 
% A function to compute the maximum possible variance of a timeseries, given a certain mean M, lower measurement bound MIN, upper measurement bound MAX and length of timeseries n.
% 
% Usage
% 
% maximumVAR(M, MIN, MAX, n)
% Arguments
% 
% M	
% mean
% 
% MIN	
% lower bound of measurements
% 
% MAX	
% upper bound of measurements
% 
% n	
% length of time series
% 
% Value
% 
% maximum possible variance
% 
% Author(s)
% 
% Merijn Mestdagh
% 
% Examples
% 
% M=5
% MIN=0
% MAX=10
% n=100
% y=maximumVAR(M,MIN,MAX,n)

    %maximum given the mean M
    %extreme cases
    if (M==MIN || M==MAX)
        mv=0;

    %normal case
    else
        if abs(MIN)>abs(MAX) %mirror for special cases like MIN=-INF
            MINt=-MAX;
            MAX=-MIN;
            MIN=MINt;            
            M=-M;
        end
        nMax=floor((n*M-n*MIN)/(MAX-MIN)); %compute nb
        nMin=n-1-nMax; %compute na
        if nMax==0
            MAX=0;
        end
        m=n*M-nMin*MIN-nMax*MAX; %compute m
        mv=(nMin*(MIN-M)^2+nMax*(MAX-M)^2+(M-m)^2)/(n-1); %compute maximum variability
    end
    
end

