# Classification-Neural-Network

A classification Neural Network with customizable hyper-parameters, bundled with data proccesing functions and data loading functions. Includes logging and plotting functionality for comparison of hyper-parameters and evaluating performance of the network.

To use, being by initalizing the class and loading the data:

    ANN = Neural_Network()
    ANN.load.data(training_data, training_labels, validation_data, validation_labels, test_data, test_labels)

    Followed by setting all the hyper parameters. We will use MNIST as an example below:
    (for more info on input parameters, see docstrings for each method)

    ANN.set_classes(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    ANN.set_hidden_layers([30])  # list of hidden layers sizes, here its one layer with 30 neurons
    ANN.set_batch_size(1)  # amount of examples per mini-batch
    ANN.set_learning_rate(0.1, "fixed")  # learning rate, and which learning algorithm to use
    ANN.set_regularization(1)  # lambda value for L2 regularization (0 to turn it off)
    ANN.set_no_improvement_offset(10)  # If the network doesnt improve in 10 epochs, it terminates
    ANN.set_cost(nn.ann.Cross_Entropy_Cost)  # the cost, or loss function
    ANN.set_cost(nn.ann.Sigmoid_Logistic)  # the logistic function to be used
    
    With all hyper parameters set, one can begin training the network. Here the frequency
    of applying the testing and validation set on the network has been set to 10000

    ANN.start_training(print_offset=10_000, plot_output=True)
