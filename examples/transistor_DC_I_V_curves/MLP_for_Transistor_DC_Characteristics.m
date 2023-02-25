%{
This example shows the setup of a simple MLP for modeling the nonlinear DC
characteristics (Ids vs. Vds over Vgs curves) of a FET.

The script is written to be treated as a Text User Interface for all the
operations, including:
    - Importing training and validation data
    - Processing data
    - Setting up the MLP
    - Training the MLP using Levenberg-Marquardt algorithm
    - Comparing the results of the trained model with validation data

User can treat each line of code as a button by highlighting each line and
running it.

It is recommended that the user runs the script line by line, or executes
selections or chunks of code to see what the script is doing.

- Keith Ha (Feb 2023)

%}

%% Adding Search Path for MATLAB
addpath(fullfile('..', '..', 'core'));      % folder containing core Neural Net structures
addpath(fullfile('..', '..', 'tools'));      % folder containing other misc tools for data processing, signal processing

%% Data import for DC
dataFileDir = fullfile('trainingData', 'FET_Ids_vs_Vds_over_Vgs.csv');
dataTableDC = readtable(dataFileDir);

Ids_table = dataTableDC(:,'DC_IDS_i_0____');        
Ids_vec = table2array(Ids_table);       % unit: A

% Clean up/Garbage collection of the variables that won't used in the rest
% of the script
clear('dataFileDir', 'Ids_table');

%% Data processing and plotting
% Vds and Vgs are the input variables that should have already been defined
% for the intial simulation or measurement of the transistor.
% It is easier to recall that defined ranges of Vds and Vgs instead of the
% extracting them from the data file.
Vds = (0:0.01:5)';
Vgs = (-2:0.1:1)';

Ids = reshape(Ids_vec, length(Vds), length(Vgs));   
figure; plot(Vds,Ids);


%% Splitting data to Training Set
Vds_tr = (0:0.02:5)';
Vgs_tr = (-2:0.2:1)';
indexVds_tr = extractIndex(Vds_tr, Vds);        % extractIndex() returns the array of indexes of the keys
indexVgs_tr = extractIndex(Vgs_tr, Vgs);

Ids_tr = Ids(indexVds_tr, indexVgs_tr);
figure; plot(Vds_tr, Ids_tr);
xlabel('Vds'); ylabel('Ids');
title('Training Data');

% Reshape data to 1D vec for plugging into the training function
[Vgs_tr_grid, Vds_tr_grid] = meshgrid(Vgs_tr, Vds_tr);

Vgs_tr_vec = reshape(Vgs_tr_grid, [], 1);
Vds_tr_vec = reshape(Vds_tr_grid, [], 1);

Ids_tr_vec = reshape(Ids_tr, [], 1);

% Compile all data into arrays
xdata = [Vgs_tr_vec, Vds_tr_vec];       % input data
numSamples = size(xdata,1); 
ddata = [Ids_tr_vec];                   % output data

data = [xdata ddata];

%% Splitting data into Validation Set
Vds_vl = (0.01:0.02:4.99)';
Vgs_vl = (0.1:0.2:0.9)';

indexVds_vl = extractIndex(Vds_vl, Vds);
indexVgs_vl = extractIndex(Vgs_vl, Vgs);

Ids_vl = Ids(indexVds_vl, indexVgs_vl);
figure; plot(Vds_vl, Ids_vl);
xlabel('Vds'); ylabel('Ids');
title('Validation data');

% Reshape validation data to 1D vec for later validation use
[Vgs_vl_grid, Vds_vl_grid] = meshgrid(Vgs_vl, Vds_vl);

Vgs_vl_vec = reshape(Vgs_vl_grid, [], 1);
Vds_vl_vec = reshape(Vds_vl_grid, [], 1);

xdata_vl = [Vgs_vl_vec, Vds_vl_vec];


%% Neural Net setup
% Reset the Neural Net if there's any existing
clear('numInputNeurons', 'numNeuronsHLayers', 'numOutputNeurons', ...
    'lengthWVec', 'mlp', ...
    'neuronStruct');

% Neural Net settings
% Multiple Hidden Layers can be added by making numNeuronsHLayers an array
numInputNeurons = 2;    % Vgs, Vds
numNeuronsHLayers(1) = 10;
numNeuronsHLayers(2) = 10;
numOutputNeurons = 1;   % Ids

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

% Intialize weights and biases
w0_vec = randn(lengthWVec,1);

% Compile all the structure settings into one struct to easily throw into
% the MLP function
neuronStruct.numInputNeurons = numInputNeurons;
neuronStruct.numNeuronsHLayers = numNeuronsHLayers;
neuronStruct.numOutputNeurons = numOutputNeurons;

% Define the function handle for the Neural Net function
% tanh() is used as the activation function, as denoted in the MLP function
mlp = @(w_vec, x) MLP_tanh(w_vec, x, neuronStruct);

% This is how the Neural Net is used
[y0_tr_vec, wlayers] = mlp(w0_vec, xdata); 

% y0_tr_vec is the 1D vec of values of the y output of the MLP function
% Although Ids is a 2D variable (w.r.t Vds and Vgs), the MLP function is
% designed to return 1D vec, which is reshaped from the 2D data, to work
% with training function below

% w0_vec is the 1D vec that contains all the weights and biases of the MLP,
% and it also has to be 1D to work with the training function

% wlayers is the struct that contains all the weights and biases of the MLP
% in the conventional form that corresponds to neuron connections in the
% MLP structure.
% wlayers is simply mapped from the 1D w0_vec inside the MLP function to
% organize the weights to their corresponding neurons.


%% Training algo (Levenberg-Marquardt)
% Use lsqcurvefit() from Optimization Toolbox for training

options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt', 'Display', 'iter-detailed');
% options.MaxFunctionEvaluations = lengthWVec*1000;
lb = [];        % no lower bound or upper bound
ub = [];

[wTr_vec,resnorm,residual,exitflag,output] = lsqcurvefit(mlp, w0_vec, xdata, ddata, lb, ub, options);


% Saving training results and all imported data
save(fullfile('matData', 'results.mat'));


%% Training results
[y_md_vec, wlayers] = mlp(wTr_vec, xdata);       % y_md stands for "y values of model"

% The MLP function returns a 1D vec of the model values, have to reshape to
% 2D array
% The values returned by the model has the range as the training data
Vds_md = Vds_tr;
Vgs_md = Vgs_tr;
Ids_md = reshape(y_md_vec, length(Vds_tr), length(Vgs_tr));
figure; plot(Vds_md, Ids_md);
hold on; plot(Vds_tr, Ids_tr, 'o');
title('Model vs. Training Data');


%% Plotting training results and comparing with Validation Set
[y_mdvl_vec, ~] = mlp(wTr_vec, xdata_vl);

Ids_mdvl = reshape(y_mdvl_vec, length(Vds_vl), length(Vgs_vl));
figure; plot(Vds_vl, Ids_mdvl);
hold on; plot(Vds_vl, Ids_vl, 'o');
title('Model vs. Validation Data');

%% Or simply use all the data, including both Training and Validation Set, and plot for comparison
[Vgs_grid, Vds_grid] = meshgrid(Vgs, Vds);

Vgs_vec = reshape(Vgs_grid, [], 1);
Vds_vec = reshape(Vds_grid, [], 1);

xdata_all = [Vgs_vec, Vds_vec];

[y_md_vec, ~] = mlp(wTr_vec, xdata_all);

Ids_md = reshape(y_md_vec, length(Vds), length(Vgs));
figure; plot(Vds, Ids_md);
hold on; plot(Vds, Ids);
title('Model vs. All Data');




