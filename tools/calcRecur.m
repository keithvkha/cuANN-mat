function [yAR, yAR_delay] = calcRecur(func, w_vec, Ny, xFIR)
    % This function is for calculating output variable that is dependent on 
    % its previous values, i.e. feedback, based on the Yule-Walker formulation
    % of Auto-Regregressive (AR) functions [1]

    % [1] https://en.wikipedia.org/wiki/Autoregressive_model

    % func is the function handler in which 
    % y = func(y[n-1],y[n-2],.., y[n-Ny], x[n], x[n-1], x[n-2],... x[n-Nx]

    % Ny is the order of feedback of y

    % xFIR = [ x[n], x[n-1], x[n-2], ..., x[n-Nx] ] = [x x_delay]

    %% Getting some information from the arguments first
    numSamples = size(xFIR,1);

    % Init yAR and yAR_delay as NaN because they are undefined for t = 0
    % yAR_delay will be determinted by feedback of yAR

    yAR = NaN(numSamples,1);
    yAR_delay = NaN(numSamples, Ny);
    yAR_delay(1,:) = zeros(1,Ny);        % Intializing the intial conditions to be 0 (i.e. the system is at rest before t=0)

    for n = 1:numSamples
        for k = 1:Ny
            yAR_delay(:,k) = delayshift(yAR,k);
        end
        %     xAR(n,:) = [xt(n) xt_delay(n,:) yAR_delay(n,:)];
        xAR = [xFIR yAR_delay];
        yAR(n) = func(w_vec, xAR(n,:));
    end
end