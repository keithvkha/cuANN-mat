%{
Adjoint Neural Network is a special Neural Net that contains 2 parts: the
conventional original feedforward MLP, and the Adjoint associated with the
original MLP.
The feedforward MLP is simply the basic, standard Neural Net that models
the relationship between the output and input variables: y = f(x)
The Adjoint part models the derivative of output w.r.t the input variables:
dy/dx = df(x)/dx, based on the mathematical relationship y = f(x)
determined by the original MLP
For more information and theoretical derivation of the Adjoint Neural Net,
please refer to the reference [1].

[1] Jianjun Xu, M. C. E. Yagoub, Runtao Ding and Qi Jun Zhang, 
"Exact adjoint sensitivity analysis for neural-based microwave modeling and design," 
in IEEE Transactions on Microwave Theory and Techniques, vol. 51, no. 1, pp. 226-237, Jan. 2003, doi: 10.1109/TMTT.2002.806910.

The function is designed to work with MATLAB's native lsqcurvefit() for
training the MLP. 
The parameters of this function:
    - w_vec: the vector containing all the weights and biases that are to
    be determined/optimized by the training algorithms
    - x: 2D matrix of input variables
    - neuronStruct: struct that contains the settings of the MLP structure
    - adjointStruct: struct that contains the settings of the Adjoint part

- Keith Ha, Carleton University, Feb 2023
%}

function [y, wlayers, ajwlayers, hlayers] = MLP_Adjoint_tanh(w_vec, x, neuronStruct, adjointStruct)
    %% README
    % This function sets up both the Original MLP and the Adjoint MLP
    % together for a combined training of both MLPs
    
    % w_vec is the vector that contains all the weights and biases that
    % will be trained by lsqcurvefit() Leveberg-Marquardt, and will
    % distributed to wlayers, which will be the struct that contains all 
    % the weights and biases from the Original MLP.

    % xdata is the input for computing EDNs (element derivative neurons)

    % xDeriv is the input to the Adjoint MLP for selecting which adjoint
    % outputs, yDeriv, to output.
    % xDeriv can be a row vector, e.g. [... 0 0 1 0 0 ...], where the j-th
    % element = 1 selects the x(j) for computing 
    % [dy(1)/dx(j) dy(2)/dx(j) ... dy(Nx)/dx(j)]. 

    
    
    %% init
    numInputNeurons = neuronStruct.numInputNeurons;
    numNeuronsHLayers = neuronStruct.numNeuronsHLayers;
    numOutputNeurons = neuronStruct.numOutputNeurons;
    mlpOutputSel = neuronStruct.outputSelections;
    
    % numNeuronsHLayers(hl) is an array that stores the number of hidden
    % neurons in layer hl
    numHiddenLayers = length(numNeuronsHLayers);
    numWLayers = numHiddenLayers+1;     % WLayer is the layer of weights in between the neuron layers
    numSamples = size(x,1);

    %% Weights setup
    % First, distribute the initialized weights in w_vec to their
    % corresponding layers and neurons by traversing through w_vec.
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

    %% Adjoint weights setup
    awlg = 1;        % index for adjoint wlayers, g stands for global var
    for wlg = 1:numWLayers
        ajwlayers(awlg).w =  wlayers(numWLayers-wlg+1).w(:,1:end-1);    % excluding the biases
        awlg = awlg+1;
    end

    %% Adjoint derivative selection
    xDeriv_mat = adjointStruct.xDeriv_mat;
    adjointOutputSel = adjointStruct.outputSelections;

    %% MLP Neurons setup
    % Original MLP output neurons
    yMLP = outputNeurons();
    
    yMLP_out = yMLP .* mlpOutputSel;
    % this element-wise multiplication is equivalent to selecting which
    % columns in yMLP to output
    % e.g.
    % [yMLP(:,1) yMLP(:,2) yMLP(:,3) yMLP(:,4)].*[1 1 0 0] 
    % =  [yMLP(:,1) yMLP(:,2) 0 0]
    % the unused output will be output as 0
    

    %% Adjoint Neurons setup
    yAdjoint_out = [];
    for iDeriv = 1:length(xDeriv_mat)
        xDeriv_g = xDeriv_mat(iDeriv,:);
        yAdjointCur = outputAdjointNeurons(xDeriv_g);
        yAdjointCur_out = yAdjointCur .* adjointOutputSel;

        yAdjoint_out = [yAdjoint_out yAdjointCur_out];
    end
    
    %% Function output y
    % y includes both Original output neurons and 
    % the Adjoint output neurons selected by xDeriv
    y = [yMLP_out yAdjoint_out];  
    



    %% Original MLP functions
    function outputMLP = outputNeurons()
        hl = numHiddenLayers;       % hl stands for "hidden neuron layer", is index for hiddenLayers
        wl = numWLayers;            % wl stands for "weight layer", is index for weightLayers
        
        % The neurons in the last hiddenLayer is defined by the the
        % hiddenNeurons in the prev layer; thus, hl-1
        z_last = hiddenNeurons(hl-1);

        %{
        [y(:, 1) y(:,2) ... y(:,numOutputNeurons) = [z(:,1) z(:,2) ... z(:,numHiddenNeurons) 1] * [w(1,1) w(2,1) ... w(numOutputNeurons,1)]
                                                                                                  [w(1,2) w(2,2) ... w(numOutputNeurons,2)] 
                                                                                                  [              ...                      ]
                                                                                                  [w(1,numHiddenNeurons) w(2,numHiddenNeurons) ... w(numOutputNeurons,numHiddenNeurons)]
                                                                                          bias -> [w(1,numHiddenNeurons+1) w(2,numHiddenNeurons+1) ... w(numOutputNeurons, numHiddenNeurons+1)]
        %}
        outputMLP = [z_last ones(numSamples,1)] * wlayers(wl).w';
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
            % Once reached the inputLayer, 
            % gamma = inputNeurons() * weights + bias
            gamma = [x ones(numSamples,1)] * wlayers(wl).w';

        else            % This is the recurrence relation of the recursion

            z_prev = hiddenNeurons(hl-1);

            gamma = [z_prev ones(numSamples,1)] * wlayers(wl).w';

        end
        
        z_cur = tanh(gamma);
        
        hlayers(hl+1).z = z_cur;
        hlayers(hl+1).gamma = gamma;
    end
    
    %% Adjoint functions
    function outputAdjoint = outputAdjointNeurons(xDeriv)
        ahl = numHiddenLayers;      % ahl stands for "adjoint hidden neuron layer", is index for hiddenLayers
        awl = numWLayers;           % awl stands for "adjoint weight layer", is index for weightLayers
        
        % The neurons in the last hiddenLayer is defined by the the
        % hiddenNeurons in the prev layer; thus, hl-1
        zDeriv = hiddenAdjointNeurons(ahl-1, xDeriv);

        %{
        [yDeriv(:, 1) yDeriv(:,2) ... yDeriv(:,numOutputNeurons) = [zDeriv(:,1) zDeriv(:,2) ... zDeriv(:,numHiddenNeurons)] * [w(1,1) w(1,2) ... w(1, numHiddenNeurons)]
                                                                                                                              [w(2,1) w(2,2) ... w(2, numHiddenNeurons)] 
                                                                                                                              [              ...                      ]
                                                                                                                              [w(numAdjointOutputNeurons,1) w(numAdjointOutputNeurons,2) ... w(numAdjointOutputNeurons,numHiddenNeurons)]                                                                                                       
        %}
        outputAdjoint = zDeriv * ajwlayers(awl).w;        
    end
    
    function zDeriv_cur = hiddenAdjointNeurons(ahl, xDeriv)
        % is the same as EDNs, which stands for Elemenet Derivative
        % Neurons, which are the hidden neurons in Adjoint NN

        % index for adjointWeightLayer
        awl = ahl+1;
        
        % index for hiddenLayer from Original MLP
        hl = numHiddenLayers-ahl+1;     % hiddenLayer position should be diagonal to 
                                        % adjointHiddenLayer position

        if (ahl == 0)     % This is the termination condition for the recursion
            % Once reached the inputLayer, use the inputNeurons x instead
            % of z_prev
            alpha =  xDeriv * ajwlayers(awl).w;     % alpha represents for adjointGamma

        else            % This is the recurrence relation of the recursion

            zDeriv_prev = hiddenAdjointNeurons(ahl-1, xDeriv);

            alpha = zDeriv_prev * ajwlayers(awl).w;

        end

        zDeriv_cur = (1 - tanh(hlayers(hl-1).gamma).^2) .* alpha;     % dtanh(x)_dx = 1-tanh(x)^2
    end
    
end