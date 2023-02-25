function x_delay = delayshift(x,K)
    % x_delay = x[n-K]
    % This function delays x[n] by K samples by shifting the elements of x
    % by K positions, and zero-pad the beginning

    % This function is meant for shifting 1D vector
    % For shifting a matrix, please refer to delayshiftmat(), which does
    % the shifting with a "shift operator matrix"

    % x is a column vector
    % x_delay is a column vector
    
    if(~exist('K', 'var'))
        K = 0;
    end
    
    if(size(x,1) > 1)       % x is column vector
        x_delay = [zeros(K,1); x(1:length(x)-K)];
    elseif(size(x,2) > 1)   % x is row vector
        x_delay = [zeros(1,K) x(1:length(x)-K)];
    end

end