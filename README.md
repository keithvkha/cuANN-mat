# cuANN-mat

cuANN-mat is an open-source package for customizing and prototyping basic and special Artificial Neural Networks (ANN) in MATLAB for applications in RF/Microwave and Computational Physics.

This project is motivated by the need to customize conventional ANNs for specialized regression/curve-fitting problems in RF/Microwave devices, circuits, and systems with high order of nonlinearity and memory. 

The code is developed in the style of functional programming to avoid any object-oriented abstraction so that user can easily customize the ANN for their regression problems in RF/Microwave and Physics.

## Installation
MATLAB is required to run the files in the package.

Required toolboxes:
- Optimization Toolbox

Additional toolboxes that can be used:
- Signal Processing Toolbox

This package is originally written in MATLAB 2022. Most functions used in this package should be compatible with MATLAB 2013 and after. To add more compatibility of this package with older versions of MATLAB, please send me suggestions and/or pull requests to add to the repo. 

## Usage
This package contains examples on how to use the customizable ANN core of the package. Currently, the package contains 3 examples:
 - Basic MLP for modeling the nonlinear DC characteristics of a transistor.
 - Adjoint Neural Network (AdjointNN) for modeling the DC characteristics and Y-parameters of a transistor.
 - Recurrent Neural Network (RNN) for modeling a digital filter with memory.

User can follow the documentation and comments in the examples to run the code.

For customizing the Neural Networks, user can edit the MLP functions, or create new Neural Network functions in "core" folder.

The package also utilizes MATLAB's comprehensive support and documentation for Statistics, Signal Processing, and Optimization algorithms. User can refer to MATLAB's documentation for specific functions used in the scripts.

## ANN Structure
The ANNs in this package are written as functions in the files contained in the "core" folder. User can use these functions as templates for customizing and prototyping their ANNs.

## ANN Training
The training of ANN in this package utilizes least-square algorithms from MATLAB's Optimization Toolbox. This is shown in the scripts the "examples" folder.


## Exporting the ANN model
The package also contains a script in the "utilities" folder for exporting the MLP model equation as a string with the input variables, and the numerical values of the weights and biases associated with the input and hidden neurons. 


## Contributing
Please send a pull request if you wish to contribute to this project. 


## License
GNU GPL

## References
[1] Q. J. Zhang, K. C. Gupta, "Neural Networks for RF and Microwave Design", Artech House, July 2000.

[2] Qi-Jun Zhang, K. C. Gupta and V. K. Devabhaktuni, "Artificial neural networks for RF and microwave design - from theory to practice," in IEEE Transactions on Microwave Theory and Techniques, vol. 51, no. 4, pp. 1339-1350, April 2003, doi: 10.1109/TMTT.2003.809179.

[3] Jianjun Xu, M. C. E. Yagoub, Runtao Ding and Qi Jun Zhang, "Exact adjoint sensitivity analysis for neural-based microwave modeling and design," in IEEE Transactions on Microwave Theory and Techniques, vol. 51, no. 1, pp. 226-237, Jan. 2003, doi: 10.1109/TMTT.2002.806910.

