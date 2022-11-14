# Classification-Neural-Network

A classification Neural Network with customizable hyper-parameters, bundled with data proccesing functions and data loading functions. Includes logging and plotting functionality for comparison of hyper-parameters and evaluating performance of the network.

To use, begin by initalizing the class and loading the data:
    import neural_network as nn
    
    ANN = nn.Neural_Network()
    
    # Load the data into the network
    # You can split up your data into sets and labels. TXT and CSV formats are allowed.
    nn.data_loader.split_data_set(input_data=data, label_index=0, training_set_size=0.75, validation_set_size=0.1)
    # Loads data into the model
    ANN.load.data(training_data, training_labels, validation_data, validation_labels, test_data, test_labels)
    

    # Setting up all the hyper parameters. We will use MNIST as an example below:
    # (for more info on input parameters, see docstrings for each method)

    ANN.set_classes(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)  # Each input parameter is a label, for MNIST it's numbers 0 to 9
    ANN.set_hidden_layers([30])  # list of hidden layers sizes, here its one layer with 30 neurons. Input and output layers are automatically 
                                 # calculated from data and classes
    ANN.set_batch_size(1)  # amount of examples per mini-batch
    ANN.set_learning_rate(0.1, "fixed")  # learning rate, and which learning algorithm to use, in this case fixed
    ANN.set_regularization(1)  # lambda value for L2 regularization (0 to turn it off)
    ANN.set_no_improvement_offset(10)  # If the network doesnt improve in 10 epochs, it terminates
    ANN.set_cost(nn.ann.Cross_Entropy_Cost)  # the cost, or loss function
    ANN.set_cost(nn.ann.Sigmoid_Logistic)  # the logistic function to be used
    
    # With all hyper parameters set, one can begin training the network. Here the frequency
    # of applying the testing and validation set on the network has been set to 10'000,
    # and a plot of performance will be provided at the end of training.

    ANN.start_training(print_offset=10_000, plot_output=True)
# TODO
## High Priority
- Change mini-batch into full-matrix matrix based approach for much faster computation
- BLAS optimize all linear algebra operations
- Make code Cython compatible for faster computation
## Maybe will happen
- Saving and Loading models
## Extra Options that have a very low chance of the ever happening
- Logistic functions per layer
- Adding more learning rate algorithms
- Adding more logistic and cost functions
