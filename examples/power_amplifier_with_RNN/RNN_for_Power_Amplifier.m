%{
This example shows the setup of an RNN for modeling a Nonlinear Power Amplifier (PA).

In this repo, there is no "core" RNN. The RNN is simply an MLP with a feedback loop from MLP's output.
Therefore, this example shows how to setup the feedback and the RNN from the core MLP.

The script is written to be treated as a Text User Interface for all the
operations, including:
    - Importing the data of time-domain signals, input excitation and
    output response of the PA, for training the RNN
    - Processing data
    - Setting up the RNN
    - Training the RNN using Levenberg-Marquardt algorithm in a recurrent
    feedback loop

User can treat each line of code as a button by highlighting each line and
running it.

It is recommended that the user runs the script line by line, or executes
selections or chunks of code to see what the script is doing.

- Keith Ha (Feb 2023)

%}

%% Adding Search Path for MATLAB
addpath(fullfile('..', '..', 'core'));      % folder containing core Neural Net structures
addpath(fullfile('..', '..', 'tools'));      % folder containing other misc tools for data processing, signal processing

%% Import data
fileVinReal = fullfile('trainingData', 'PA_VinPA_real.txt');
fileVinImag = fullfile('trainingData', 'PA_VinPA_imag.txt');

fileVoutReal = fullfile('trainingData', 'PA_VoutPA_real.txt');
fileVoutImag = fullfile('trainingData', 'PA_VoutPA_imag.txt');

Vin_real_table = readtable(fileVinReal);
Vin_imag_table = readtable(fileVinImag);

Vout_real_table = readtable(fileVoutReal);
Vout_imag_table = readtable(fileVoutImag);

tn = table2array(Vin_real_table(:,'time'));
Vin_real_t = table2array(Vin_real_table(:,'real_VinPA_1__'));
Vin_imag_t = table2array(Vin_imag_table(:,'imag_VinPA_1__'));

Vout_real_t = table2array(Vout_real_table(:,'real_VoutPA_1__'));
Vout_imag_t = table2array(Vout_imag_table(:,'imag_VoutPA_1__'));


clear('fileVinReal', 'fileVoutReal', ...
     'Vin_real_table', 'Vin_imag_table', ...
     'Vout_real_table', 'Vout_imag_table');


%% Plotting the signals
Ts = tn(2) - tn(1);
Fs = 1/Ts;

figure; 
subplot(2,1,1);
plot(tn, Vin_real_t);
hold on; plot(tn, Vout_real_t);
xlabel('time (sec)');
ylabel('In-phase');
legend('Vin', 'Vout');

subplot(2,1,2);
plot(tn, Vin_imag_t);
hold on; plot(tn, Vout_imag_t);
xlabel('time (sec)');
ylabel('Quadrature');
legend('Vin', 'Vout');


%% Memory order of the RNN system
Nx = 0;     % filter order of input
Ny = 2;     % filter order of output

% User is encouraged to play with Nx and Ny to see how changing the memory
% order of the RNN can affect its convergence towards a good fit with the
% training data

% For this PA, if Ny = 0, i.e. memoryless, the training error never
% converges towards 0.001, which is the target training error in this case.
% If Ny = 2, which is the default choice in this example, the training
% error converges towards the 0.001 training error.
% Higher feedback memory order can cause more difficult traning.

%% Setup input data
xtIQ = [Vin_real_t Vin_imag_t];
numX = size(xtIQ,2);

xtIQ_delay = NaN(size(xtIQ,1), Nx*size(xtIQ,2));
ix = 1;
for k = 1:Nx
    fprintf('ix = %i:%i\n', ix, ix+numX-1);
    xtIQ_delay(:, ix:ix+numX-1) = delayshift(xtIQ, k);
    ix = ix + numX;
end

% Put the data into a big table for visualization
xtdata = [tn xtIQ xtIQ_delay];


%% Setup core MLP
% Reset the Neural Net and data if there's any existing
clear('numInputNeurons', 'numNeuronsHLayers', 'numOutputNeurons', ...
    'lengthWVec', 'mlp', ...
    'neuronStruct', 'adjointStruct', ...
    'w0_vec', 'wTr_vec', 'wlayers', ...
    'xdata', 'ddata', 'y0', 'y0_delay', 'y0data', 'yt_md', 'yt_md_delay', ...
    'options', 'epoch', 'resnorm', 'residual', 'output');

numInputNeurons = numX*(1 + Nx + Ny);    % [x[n] x[n-1] x[n-2] ... x[n-Nx] y[n-1] y[n-2] ... y[n-Ny]]
numNeuronsHLayers(1) = 20;
numOutputNeurons = 2;   % yI[n] yQ[n]

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

[y0, y0_delay] = calcRecur(mlp, w0_vec, Ny, [xtIQ xtIQ_delay], numOutputNeurons);

% calcRecur() is a user-defined function in "tools" for calculating the
% output variable that is dependent on its previous values 

%% Setup output = h(n)
ddata = [Vout_real_t, Vout_imag_t];
alldata = [tn xtIQ xtIQ_delay y0_delay ddata];

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
xdata = [xtIQ xtIQ_delay y0_delay];     % Init xdata for the first lsqcurvefit() iteration

epoch = 0;
resnorm = Inf;

%% Start training Recurrent LM by highlighting and running the code below
fprintf('\n Start training Recurrent LM... \n');
while (epoch <= epochMax) && (resnorm >= resnormTarget)
    [wTr_vec,resnorm,residual,exitflag,output] = lsqcurvefit(mlp, wTr_vec, xdata, ddata, lb, ub, options);

    [yt_md, yt_md_delay] = calcRecur(mlp, wTr_vec, Ny, [xtIQ xtIQ_delay], numOutputNeurons);
    xdata = [xtIQ xtIQ_delay yt_md_delay];

    fprintf('epoch = %d, ', epoch);
    fprintf('resnorm = %.4f\n', resnorm);

    epoch = epoch + 1;
    
    % Additional code for storing the change in the weights and
    % trainingErrors
    [~,wlayers] = mlp(wTr_vec, xdata);
    wEpochs(epoch).wlayers = wlayers;
    trErrors(epoch) = resnorm;
end

%% Comparing the trained model vs. original data
[yt_md, wlayers] = mlp(wTr_vec,xdata);
Vout_real_md = yt_md(:,1);
Vout_imag_md = yt_md(:,2);

figure; title('Fitted RNN Model');
subplot(2,1,1);
plot(tn, Vin_real_t);
hold on; 
plot(tn, Vout_real_t);
plot(tn, Vout_real_md, '--');
xlabel('time (sec)');
ylabel('In-phase');
legend('Vin (original)', 'Vout (original)', 'Vout (Neural Net)');

subplot(2,1,2);
plot(tn, Vin_imag_t);
hold on;
plot(tn, Vout_imag_t);
plot(tn, Vout_imag_md, '--');
xlabel('time (sec)');
ylabel('Quadrature');
legend('Vin', 'Vout');
legend('Vin (original)', 'Vout (original)', 'Vout (RNN)');

save(fullfile('matData', 'PA_RNN.mat'));












