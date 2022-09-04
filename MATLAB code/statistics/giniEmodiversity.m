function [ G ] = giniEmodiversity( X,limit )
%GINIEMODIVERSITY  1 person

% https://quantdev.ssri.psu.edu/sites/qdev/files/Emodiversity_JoG_Tutorial_2017-01-01.html

m=size(X,2);
for i =1:m
    F(1,i)=length(find(X(:,i)>=limit));
end

C=sort(F);
giniReal=2*C*((1:m)')/(m*sum(C))-(m+1)/m;
G=giniReal;

if isnan(G)
    warning('NaN Gini');
end

end

