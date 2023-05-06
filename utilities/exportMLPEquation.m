%{
This script is used to export the string of the MLP model equation based on
the model's weights and biases after training the model.

User can specify the string of the input variables. The output string
of this script will be the MLP function in terms of the input variables the
numerical values of the weights and biases associated with the neurons.

The activation function used in this script is tanh(). 
User can change the activation function according to their ANN model by
replacing the "tanh(%s)" string with the string of the activiation function
For example,
For sigmoid logistic function: "1/(1 + exp(-%s))"

%}

%% Generate equations for 3-layer MLP or Deep MLP
% Reset variables if there's any existing
clear('xStr', 'zStr', 'yStr');

% Load data and trained model
dataFileDir = fullfile('..','examples','transistor_DC_I_V_curves','matData', 'results.mat');
load(dataFileDir);

% Assign strings for the input variables
xStr(1) = "_v1";
xStr(2) = "_v2";

numWeightLayers = length(wlayers);

if numWeightLayers == 2     % 3-layer MLP
    % Equation str for weightLayer 1
    wl = 1;
    numHiddenNeuronsLayer1 = size(wlayers(wl).w, 1);
    numInputNeuronsAndBias = size(wlayers(wl).w, 2);
    for i = 1:numHiddenNeuronsLayer1
        gammaStr(i) = "";
        zStr(i) = "";
        for j = 1:numInputNeuronsAndBias
            if (j == numInputNeuronsAndBias)
                gammaStr(i) = gammaStr(i) + sprintf("(%.10f)",wlayers(wl).w(i,j));      % bias
            else
                gammaStr(i) = gammaStr(i) + sprintf("(%.10f)*%s + ", wlayers(wl).w(i,j), xStr(j));    % weights
            end
            zStr(i) = sprintf("tanh(%s)", gammaStr(i));
        end
    end

    % Equation for the last weightLayer (outputLayer)
    wl = numWeightLayers;
    numOutputNeurons = size(wlayers(wl).w, 1);
    numNeuronsLastHLayerAndBias = size(wlayers(wl).w,2);

    for i = 1:numOutputNeurons
        yStr(i) = "";
        for j = 1:numNeuronsLastHLayerAndBias
            if(j == numNeuronsLastHLayerAndBias)
                yStr(i) = yStr(i) + sprintf("(%.10f)",wlayers(wl).w(i,j));
            else
                yStr(i) = yStr(i) + sprintf("(%.10f)*%s + ", wlayers(wl).w(i,j), zStr(j));
            end
        end
    end

elseif numWeightLayers >  2     % Deep MLP
    % Equation str for weightLayer 1
    wl = 1;
    numHiddenNeuronsLayer1 = size(wlayers(wl).w, 1);
    numInputNeuronsAndBias = size(wlayers(wl).w, 2);
    for i = 1:numHiddenNeuronsLayer1
        gammaStr(i) = "";
        zStr(i) = "";
        for j = 1:numInputNeuronsAndBias
            if (j == numInputNeuronsAndBias)
                gammaStr(i) = gammaStr(i) + sprintf("(%.10f)",wlayers(wl).w(i,j));      % bias
            else
                gammaStr(i) = gammaStr(i) + sprintf("(%.10f)*%s + ", wlayers(wl).w(i,j), xStr(j));    % weights
            end
            zStr(i) = sprintf("tanh(%s)", gammaStr(i));
        end
    end


    % Equation for the weightLayers between the Hidden Layers
    for wl = 2:numWeightLayers-1        % not including the lat weightLayer (between the last hiddenLayer and outputLayer)
        xStr = zStr;
        numNeuronsNextHLayer = size(wlayers(wl).w,1);
        numNeuronsPrevHLayer = size(wlayers(wl).w,2);

        for i = 1:numNeuronsNextHLayer
            gammaStr(i) = "";
            zStr(i) = "";
            for j = 1:numNeuronsPrevHLayer
                if (j == numNeuronsPrevHLayer)
                    gammaStr(i) = gammaStr(i) + sprintf("(%.10f)",wlayers(wl).w(i,j));      % bias
                else
                    gammaStr(i) = gammaStr(i) + sprintf("(%.10f)*%s + ", wlayers(wl).w(i,j), xStr(j));    % weights
                end
                zStr(i) = sprintf("tanh(%s)", gammaStr(i));
            end
        end
    end

    % Equation for the last weightLayer (outputLayer)
    wl = numWeightLayers;
    numOutputNeurons = size(wlayers(wl).w, 1);
    numNeuronsLastHLayerAndBias = size(wlayers(wl).w,2);

    for i = 1:numOutputNeurons
        yStr(i) = "";
        for j = 1:numNeuronsLastHLayerAndBias
            if(j == numNeuronsLastHLayerAndBias)
                yStr(i) = yStr(i) + sprintf("(%.10f)",wlayers(wl).w(i,j));
            else
                yStr(i) = yStr(i) + sprintf("(%.10f)*%s + ", wlayers(wl).w(i,j), zStr(j));
            end
        end
    end

end


%% Copy equation to clipboard
fprintf("y = %s", yStr(1));
fprintf("\n");
clipboard('copy', yStr(1));





