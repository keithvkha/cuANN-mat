%{
This example shows the setup of an RNN for modeling an IIR digital filter.

In this repo, there is no "core" RNN. The RNN is simply an MLP, which is a
core, with feedback loop from MLP's output. Therefore, this example shows
how to setup the feedback and the RNN from the core MLP.

This example also sets up a digital Butterworth filter by using MATLAB's 
Signal Processing Toolbox to generate training data for the RNN instead of 
importing data from external resources.

The script is written to be treated as a Text User Interface for all the
operations, including:
    - Creating the Butterworth filter to generate training data
    - Processing data
    - Setting up the RNN
    - Training the RNN using Levenberg-Marquardt algorithm

User can treat each line of code as a button by highlighting each line and
running it.

It is recommended that the user runs the script line by line, or executes
selections or chunks of code to see what the script is doing.

- Keith Ha (Feb 2023)

%}

%% Adding Search Path for MATLAB
addpath(fullfile('..', '..', 'core'));      % folder containing core Neural Net structures
addpath(fullfile('..', '..', 'tools'));      % folder containing other misc tools for data processing, signal processing

%% Create a digital Butterworth filter to generate training data
N = 4;      % Filter memory order
Fs = 10e3;  % Sampling freq
fc = 3000;  % Centre freq

[bz, az] = butter(N, 2*(fc/Fs), 'high');

omega = (-pi:2*pi/1000:pi-2*pi/1000)';      % normalized freq
HHPz = freqz(bz, az, omega);                % Frequency response
figure; plot(omega/pi, abs(HHPz));

[hHPz_impz, n] = impz(bz, az);              % Time-domain impulse response
figure; plot(n, hHPz_impz);
title('Impulse response');
xlabel('n (samples)');
ylabel('h(n)');


%% Memory order of the RNN system
Nx = 4;     % filter order of input
Ny = 4;     % filter order of output

% User is encouraged to play with Nx and Ny to see how changing the memory
% order of the RNN can affect its convergence towards the original impulse
% response of the original IIR filter

%% Setup input data
tn = n;                 % discrete time
xt = discDelta(n);      % Use discrete Dirac delta function as the input

% Delay the input by using delayshift(), which is user-defined function as
% part of the "tools" folder
xt_delay = NaN(length(tn), Nx);
for k = 1:Nx
    xt_delay(:,k) = delayshift(xt, k);
end

% Put into one matrix for better visualization of the data
xtdata = [tn xt xt_delay];


%% Setup core MLP
% Reset the Neural Net and data if there's any existing
clear('numInputNeurons', 'numNeuronsHLayers', 'numOutputNeurons', ...
    'lengthWVec', 'mlp', ...
    'neuronStruct', 'adjointStruct', ...
    'w0_vec', 'wTr_vec', 'wlayers', ...
    'xdata', 'ddata', 'y0', 'y0_delay', 'y0data', 'yt_md', 'yt_md_delay', ...
    'options', 'epoch', 'resnorm', 'residual', 'output');

numInputNeurons = 1 + Nx + Ny;    % [x[n] x[n-1] x[n-2] ... x[n-Nx] y[n-1] y[n-2] ... y[n-Ny]]
numNeuronsHLayers(1) = 5;
numOutputNeurons = 1;   % y[n]

% Calculate the number of weights + biases (aka parameters) based on the 
% specified number of neurons
if length(numNeuronsHLayers) == 1       % shallow/3-layer neural net
    hl = 1;
    lengthWVec = numNeuronsHLayers(1)*(numInputNeurons+1) + numOutputNeurons*(numNeuronsHLayers(hl)+1);

else                                    % Multi hidden layers
    lengthWVec = numNeuronsHLayers(1)*(numInputNeurons+1);
    for hl = 2:length(numNeuronsHLayers)
        lengthWVec = lengthWVec + numNeuronsHLayers(hl)*(numNeuronsHLayers(hl-1)+1);
    end
    lengthWVec = lengthWVec + numOutputNeurons*(numNeuronsHLayers(hl)+1);
end

w0_vec = randn(lengthWVec,1);

neuronStruct.numInputNeurons = numInputNeurons;
neuronStruct.numNeuronsHLayers = numNeuronsHLayers;
neuronStruct.numOutputNeurons = numOutputNeurons;

mlp = @(w_vec, x) MLP_tanh(w_vec, x, neuronStruct);

[y0, y0_delay] = calcRecur(mlp, w0_vec, Ny,[xt xt_delay]);

% calcRecur() is a user-defined function in "tools" for calculating the
% output variable that is dependent on its previous values 


%% Setup output = h(n)
ddata = hHPz_impz;
alldata = [tn xt xt_delay y0_delay ddata];

%% Settings for Recurrent Levenberg-Marquadt (LM variation of Backprop Through Time) 
% Customizing Matlab's native lsqcurvefit() to implement trainLM-ThroughTime
% For more information of the hyper-parameters of the LM algorithm, please
% visit MATLAB's documentation for lsqcurvefit()

options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt', 'Display', 'off');
options.MaxIterations = 1;
% options.InitDamping = 100;
% options.MaxFunctionEvaluations = lengthWVec*1000;
lb = [];        % no lower bound or upper bound
ub = [];

epochMax = 1000;
resnormTarget = 0.001;
trErrors = NaN(epochMax,1);
wEpochs(epochMax) = struct();

%% Intialize the weights and biases, and input for the recurrent feedback loop 
wTr_vec = w0_vec;
xdata = [xt xt_delay y0_delay];     % Init xdata for the first lsqcurvefit() iteration

epoch = 0;
resnorm = Inf;

%% Start training Recurrent LM by highlighting and running the code below
fprintf('\n Start training Recurrent LM... \n');
while (epoch <= epochMax) && (resnorm >= resnormTarget)
    [wTr_vec,resnorm,residual,exitflag,output] = lsqcurvefit(mlp, wTr_vec, xdata, ddata, lb, ub, options);

    [yt_md, yt_md_delay] = calcRecur(mlp, wTr_vec, Ny, [xt xt_delay]);
    xdata = [xt xt_delay yt_md_delay];

    fprintf('epoch = %d, ', epoch);
    fprintf('resnorm = %.4f\n', resnorm);

    epoch = epoch + 1;
    
    % Additional code for storing the change in the weights and
    % trainingErrors
    [~,wlayers] = mlp(wTr_vec, xdata);
    wEpochs(epoch).wlayers = wlayers;
    trErrors(epoch) = resnorm;
end

save(fullfile('matData', 'results.mat'));


%% Comparing the trained model vs. original data
[yt_md, wlayers] = mlp(wTr_vec, xdata);
figure; plot(tn, yt_md);
hold on; plot(tn, ddata, 'o');
title('Impulse response - Trained RNN vs. Original Filter');
xlabel('h(n)');
ylabel('n (samples)');
legend('trained RNN', 'original filter');





