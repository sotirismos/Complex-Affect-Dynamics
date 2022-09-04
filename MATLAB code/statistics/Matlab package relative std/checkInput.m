function [   ] = checkInput( X,MIN,MAX )
%a function the check the input of the relative variability indices

if length(find(X>MAX))>0
    error('Values found bigger than the maximum') 
elseif length(find(X<MIN))>0
    length(find(X<MIN))
    error('Values found smaller than the minimum') 
elseif MIN>MAX
    error('Error, MIN>MAX');
end


end

