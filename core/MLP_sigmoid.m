function [y, wlayers] = MLP_tanh(w_vec, x, neuronStruct)
    %% init
    numInputNeurons = neuronStruct.numInputNeurons;
    numNeuronsHLayers = neuronStruct.numNeuronsHLayers;
    numOutputNeurons = neuronStruct.numOutputNeurons;

    % numNeuronsHLayers(hl) is an array that stores the number of hidden
    % neurons in layer hl
    numHiddenLayers = length(numNeuronsHLayers);
    numWLayers = numHiddenLayers+1;     % WLayer is the layer of weights in between the neuron layers
    numSamples = size(x,1);
    
    %% Weights setup
    % First, distribute the initialized weights in w_vec to their
    % correspoding layers and neurons by traversing through w_vec.
    % The weights will be stored in a struct named wlayers, which represents, 
    % the 3-dimensional indexing of the weights


    % Setting up indexing
    v = 1;      % global index for traversing w_vec
    
    % The last weightLayer (between outputLayer and last hiddenLayer)
    hlg = numHiddenLayers;
    wlg = numWLayers;           % wlg = hlg+1
    for in = 1:numOutputNeurons
        for jn = 1:numNeuronsHLayers(hlg)+1
            wlayers(wlg).w(in,jn) = w_vec(v);
            v = v+1;
        end
    end
    
    % The weightLayer between each 2 hiddenLayers
    wlg = numWLayers-1;
    while(hlg > 1)        % while still between the hiddenLayers
        for in = 1:numNeuronsHLayers(hlg)
            for jn = 1:numNeuronsHLayers(hlg-1)+1
                wlayers(wlg).w(in,jn) = w_vec(v);
                v = v+1;
            end
        end
        hlg = hlg-1;
        wlg = wlg-1;
    end

    % After the looping through the hiddenLayers, 
    % hlg shoud = 1 (first hiddenLayer), wlg should = 1, meaning it
    % has reached between the first hiddenLayer and inputLayer
    for in = 1:numNeuronsHLayers(hlg)
        for jn = 1:numInputNeurons+1
            wlayers(wlg).w(in,jn) = w_vec(v);
            v = v+1;
        end
    end

    %% Neurons setup
    y = outputNeurons();
    
    
    function output = outputNeurons()
        hl = numHiddenLayers;    % hl stands for "hidden neuron layer", is index for hiddenLayers
        wl = numWLayers;        % wl stands for "weight layer", is index for weightLayers
        
        % The neurons in the last hiddenLayer is defined by the the
        % hiddenNeurons in the prev layer; thus, hl-1
        z = hiddenNeurons(hl-1);

        %{
        [y(:, 1) y(:,2) ... y(:,numOutputNeurons) = [z(:,1) z(:,2) ... z(:,numHiddenNeurons) 1]*[w(1,1) w(2,1) ... w(numOutputNeurons,1)]
                                                                                                [w(1,2) w(2,2) ... w(numOutputNeurons,2)] 
                                                                                                [              ...                      ]
                                                                                                [w(1,numHiddenNeurons) w(2,numHiddenNeurons) ... w(numOutputNeurons,numHiddenNeurons)]
                                                                                        bias -> [w(1,numHiddenNeurons+1) w(2,numHiddenNeurons+1) ... w(numOutputNeurons, numHiddenNeurons)]
        %}
        output = [z ones(numSamples,1)] * wlayers(wl).w';
    end
    
    
    function z_cur = hiddenNeurons(hl)
        % This function acts like a recursive function that calls itself to
        % setup prev-layer's neurons z_prev to calculate this current neuron z_cur

        % The recursion traverses through the MLP network as 
        % depth-first traversal

        % needs to do "depth-first traversal" to assign the numerical value to
        % each hidden neuron; therefore, recursion is needed for a clean
        % implementation, similar to depth-first search or sorting algos like quicksort/mergesort

        % index for weightLayer
        wl = hl+1;

        if (hl == 0)     % This is the termination condition for the recursion
            % Once reached the inputLayer, use the inputNeurons x instead
            % of z_prev
            gamma = [x ones(numSamples,1)] * wlayers(wl).w';

        else            % This is the recurrence relation of the recursion

            z_prev = hiddenNeurons(hl-1);

            gamma = [z_prev ones(numSamples,1)] * wlayers(wl).w';
        end

        z_cur = 1 ./ (1 + exp(-gamma));        % sigmoid activation function

    end
end


