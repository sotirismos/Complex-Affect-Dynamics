function [  ] = checkOutput( M,MIN,MAX )
%a function the check the output of the relative variability indices

    if M==MAX
        warning('NaN returned. Data has a mean of %f which is the maximum',M)
    elseif M==MIN
        warning('NaN returned. Data has a mean of %f which is the minimum',M)
    else
        warning('NaN returned.')
    end
end

