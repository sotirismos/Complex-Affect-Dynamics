function [ rv,mv ] = relativeSD (X,MIN,MAX)

% Description
% 
% a function to compute the relative standard deviation, the standard deviation*
% 
% Usage
% 
% relativeSD(X, MIN, MAX)
% Arguments
% 
% X	
% a vector
% 
% MIN	
% the lower bound of the measurments
% 
% MAX	
% the upper bound of the measurments
% 
% Value
% 
% rv=standard deviation*
% mv=maximum possible standard deviation
% 
% Author(s)
% 
% Merijn Mestdagh
% 
% Examples
% 
% x=[1,2,3]
% y=relativeSD(x,0,10)

    M=mean(X);
    SD=std(X);
    checkInput(X,MIN,MAX);
    n=length(X);
    mv=maximumVAR(M,MIN,MAX,n); %compute the maximum possible standard deviation given the mean
    mv=sqrt(mv);
    if mv~=0
        rv=SD/mv; %compute the relative std
    else
        rv=NaN;
        checkOutput(M,MIN,MAX);
    end
        
end

