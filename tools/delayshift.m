function x_delay = delayshift(x,K)
    % x_delay = x[n-K]
    % This function delays x[n] by K samples by shifting the elements of x
    % by K positions by using circshift(), and zero-pad up to K elements in the beginning

    % x are column vectors
    % x_delay are a column vectors
    
    if(~exist('K', 'var'))
        K = 0;
    end
    
    x_delay = circshift(x,K,1);
    for i = 1:K
        x_delay(i,:) = zeros(1,size(x,2));
    end

end