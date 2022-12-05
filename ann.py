from typing import Callable
from scipy.special import expit
import numpy as np
import matplotlib.pyplot as plt

# My own modules
import mnist_loader
import data_processing
import data_loader


class Quadratic_Cost:
    @staticmethod
    def cost(y: np.ndarray, z: np.ndarray, a: np.ndarray) -> float:
        abs_value = np.linalg.norm(a - y)
        return 0.5*np.square(abs_value)

    @staticmethod
    def delta(y: np.ndarray, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        return (a - y) * Sigmoid_Logistic.d_logistic(z)


class Cross_Entropy_Cost:
    @staticmethod
    def cost(y: np.ndarray, z: np.ndarray, a: np.ndarray) -> float:
        sum_array = y * np.log(a) + (1 - y) * np.log(1 - a)
        return -(1/len(a)) * np.sum(sum_array, axis=0)

    @staticmethod
    def delta(y: np.ndarray, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        return a - y


class Sigmoid_Logistic:
    @staticmethod
    def logistic(z: np.ndarray) -> np.ndarray:
        return expit(z)
    
    @staticmethod
    def d_logistic(z: np.ndarray) -> np.ndarray:
        sig = Sigmoid_Logistic.logistic(z)  # saves value for faster calculation
        return sig*(np.subtract(1, sig))


class Neural_Network:
    """Neural Network is a neural network able to perform classification problems. 
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
    """
    def __init__(self) -> None:
        # Layer variables
        self.hidden_layer_sizes: list[int]
        self.input_layer_size: int
        self.output_layer_size: int
        self.class_dict: dict

        # Labels
        self.training_labels: np.ndarray
        self.validation_labels: np.ndarray
        self.testing_labels: np.ndarray
        
        # Data sets
        self.training_set: np.ndarray
        self.validation_set: np.ndarray
        self.testing_set: np.ndarray

        # Feedforward and Backprop variables
        self.a: list[np.ndarray]
        self.z: list[np.ndarray]
        self.w: list[np.ndarray]
        self.b: list[np.ndarray]
        self.error: list[np.ndarray]

        # Hyper Parameters
        self.batch_size: int

        self.learning_rate: float
        self.learning_rate_mode: str
        self.learning_rate_factor: int
        self.initial_learning_rate: float

        self.no_improvement_offset: int
        self.regularization: float

        self.logistic: Callable
        self.d_logistic: Callable
        self.cost: Callable
        self.delta: Callable

        # Plotting Variables
        self.figure = None
        self.ax = None
        
        # Logging
        self.test_pass_rate_list: list[float] = []
        self.test_cost_list: list[float] = []
        self.val_pass_rate_list: list[float] = []
        self.val_cost_list: list[float] = []
        self.examples_list: list[int] = []

        # For early stopping
        self.epoch_test_pass_rate: list[float] = []
    
    # =============================== Network Methods ===================================

    def _feedforward(self, input_layer: np.ndarray) -> None:
        """Updates the activations depending on the input weights and biases. Vectorized to improve calculation speed.

        Args:
            input_layer (np.ndarray): The input data if a single example in form of a (n, 1) array, where n is the amount
            of attributes
        """
        # Calculates activation values and saves them for quick access during backpropagation
        self.a[0] = input_layer
        for l in range(1, len(self.hidden_layer_sizes)+2):  # skips input layer
            dot = np.dot(self.w[l], self.a[l-1])
            dot = dot.reshape(dot.shape[0], 1)  # changes 1-d array to 2-d array, since self.b[l] is a 2-d array
            self.z[l] = dot + self.b[l]
            self.a[l] = self.logistic(self.z[l])

    def _calculate_error(self, label) -> None:
        """Calculates dC/dz of the output layer, and then backpropagates the error through
        the network. 

        Args:
            label : The real output corresponding to the example, as a single value
        """
        
        y = self._desired_output(label)
        self.error[-1] = self.delta(y, self.z[-1], self.a[-1])  # calculates error at output layer
        # layer is reversed to be consistent with notation
        for l in reversed(range(len(self.hidden_layer_sizes)+1)):  # skips input and output layer
            self.error[l] = np.dot(self.w[l+1].T, self.error[l+1]) * self.d_logistic(self.z[l])

    def _update_delta(self) -> None:
        """Applies the error calculated by self._calculate_error to the biases and weights. Includes
        L2 regularization.
        """
        for l in range(1, len(self.hidden_layer_sizes)+2):  # skips input layer
            for j in range(len(self.error[l])):  # iterates over neuron index of current layer (just like notation)
                # self.a[l-1] is a 2 dimensional array, (8, 1) for example, but error and w are 1-d arrays, (8,), 
                # so we convert it to a 1-d array here
                a = self.a[l-1].reshape(self.a[l-1].shape[0])
                # print(f"delta:{-(self.learning_rate * a * self.error[l][j])}")
                reg_scaling = (self.learning_rate * self.regularization) / len(self.training_labels)
                self.w[l][j] = (1 - reg_scaling) * self.w[l][j] - (self.learning_rate * a * self.error[l][j]) / self.batch_size  # type: ignore
            self.b[l] = self.b[l] - (self.learning_rate * self.error[l]) / self.batch_size  # type: ignore

    def _gradient_descent(self, batch, labels) -> None:
        """Performs gradient descent on the network. Note that having higher batch sizes will increase
        speed of calculation.

        Args:
            batch: Array of examples to be iterated over before applying error. If only one is provided it acts like stochastic gradient descent
            labels: Array of the real values to each example
        """
        for input_layer, label in zip(batch, labels):  # type: ignore
            self._feedforward(input_layer)
            self._calculate_error(label)
        self._update_delta()
    
    def _desired_output(self, label):
        """Takes in a label and converts it into a vector, with the correct class set to 1,
        and the rest set 0.

        Args:
            label : The real output corresponding to the example, as a single value

        Returns:
            _type_: vectorized version of the output, in form of One-Hot Encoding
        """
        correct_index: int = self.class_dict[label]
        y = np.zeros([self.output_layer_size, 1])
        y[correct_index] = 1
        return y

    def _create_arrays(self) -> None:
        """Initializes vectors to hold all the variables needed during feedforward and backpropagation. 
        Activations, z and error are init as 0, as they get overwritten during feedforward and backpropagation.
        Bias is init as 0 for simplicity, and weights are init with a Gaussian Normal of mean 0 and standard
        deviation of 1.
        """
        # Updated weight initializations from random into gaussian normal with mean 0 and standard deviation of 1/sqrt(n)
        # where n is weights connected to a neuron

        # Creates arrays of all the variables to save, and fills them with normally distributed values
        layers: list = self.hidden_layer_sizes + [self.output_layer_size]
        # Additional element in front of self.a is for input data, and for the rest it is there to keep notation consistent
        self.b = [np.array([0, 0])] + [np.random.normal(0, 1, size=[neurons, 1]) for neurons in layers]  # type: ignore
        self.a = [np.array([0, 0])] + [np.zeros([neurons, 1]) for neurons in layers]  # type: ignore
        self.z = [np.array([0, 0])] + [np.zeros([neurons, 1]) for neurons in layers]  # type: ignore
        self.error = [np.array([0, 0])] + [np.zeros([neurons, 1]) for neurons in layers]  # type: ignore

        layers = [self.input_layer_size] + layers
        # weight notation is usually w_jk, with j being neuron index for layer L, and k being neuron index
        # for layer L-1, the same index order is implemented here.
        # Additional element in front to keep notation consistent
        # self.w = [np.array([0, 0])] + [np.random.normal(0, 1, size=[neurons, prev_neurons]) for neurons, prev_neurons in zip(layers[1:], layers)]  # type: ignore
        self.w = [np.array([0, 0])] + [np.random.normal(0, 1/np.sqrt(prev_neurons), size=[neurons, prev_neurons]) for neurons, prev_neurons in zip(layers[1:], layers)]  # type: ignore


    # =============================== Learning ===================================
    
    def start_training(self, print_offset: int = 100, plot_output=False, log_offset=-1) -> None:
        """Creates the arrays to hold all the values, then begins training through gradient descent. The learning
        will stop if enough epochs have passed without the network improving, decided by self._no_improvement.
        At each print_offset amount of examples, the network will be evaluated against the validation and testing set, 
        and the result printed in the terminal. Additionally, if logging is on, it will also evaluate the network each
        log_offset amount of examples, but not print them out. The result will be plotted on a graph when the network terminates
        if plot_output is set to True.

        Note that when either log_offset and print_offset will run, the network is tested on the validation and
        testing set, which can be computationally expensive if it is set too low. For maximum learning speed, turn both off.
        Note however, that after each epoch the network will be tested either way, as it is required for self._no_improvement.

        Args:
            print_offset (int): How many examples have to run before testing the network on the validation and testing sets
                                and output the result
            plot_output (bool, optional): If the perfomance of the network should be plotted. Defaults to False.
            log_offset (int, optional): How often the network performance should be saved, used in plotting. If -1, no additional logs will be taken. Defaults to -1.
        """
        # self._save_values will run on both offsets, so to avoid duplicated inputs log offset is omitted
        if print_offset == log_offset:
            log_offset = -1

        self._create_arrays()  # initializes the arrays
        batches = 0
        epochs = 0
        while True:
            if self._no_improvement(epochs): break
            nr_of_batches: int = int(np.floor(len(self.training_labels) / self.batch_size))  # type: ignore
            if plot_output:  # saves initial values
                self._save_values(batches)
            for i in range(nr_of_batches):   
                lower_index = i * self.batch_size
                upper_index = (i+1) * self.batch_size
                self._gradient_descent(self.training_set[lower_index:upper_index], self.training_labels[lower_index:upper_index])  # type: ignore
                self.print_training_info(batches, epochs, print_offset, plot_output)
                if plot_output and log_offset != -1:
                    if ((batches+1) * self.batch_size) % log_offset == 0:
                        self._save_values(batches)
                batches += 1
            test_pass_rate ,_ ,_ ,_ = self._save_values(batches)
            self.epoch_test_pass_rate.append(test_pass_rate)
            epochs += 1

        if plot_output:
            print("Plotting...")
            self.show_plot()

    def _no_improvement(self, epochs: int) -> bool:
        """Stops the network if no improvement has been made in a certain amount of epochs, specified by the epoch parameter.
        If learning rate has been set to variable, then instead of terminating the learning rate will be halved. Once the learning
        rate is below self.learning_rate_factor, set when creating the network, the network will terminate. 

        Args:
            epochs (int): Amount of epochs that have passed

        Returns:
            bool: True if the network should terminate, False if not
        """

        # checks last [offset] amount of epochs to see if average improvement is above zero, if it is not, return True
        if epochs < self.no_improvement_offset:  # not enough epochs have passed to see if no improvement has been made
            return False

        last_offset_index = epochs - self.no_improvement_offset
        last_offset_max = np.max(self.epoch_test_pass_rate[last_offset_index:])

        best_pass_rate = np.max(self.val_pass_rate_list)
        if last_offset_max < best_pass_rate:  # Checks if no improvement has been made
            if self.learning_rate_mode == "fixed":
                return True
            self.learning_rate = self.learning_rate / 2
            if self.learning_rate < (self.initial_learning_rate / self.learning_rate_factor):
                return True
        return False

    def print_training_info(self, batch_i: int, epoch_i: int, offset:int, plot_output: bool) -> None:
        """Evalutes the network on the validation and testing set, and prints the results"""
        if batch_i == 0:
            print(f"{'Validation':10} | {'Pass Rate':15} | {'Percent':8} | {'Examples':12} | {'Batch':12} | {'Epoch':6} || {'Testing':6} | {'Pass Rate':15} | {'Percent':8}")
        if ((batch_i+1) * self.batch_size) % (offset) == 0:
            test_pass_rate, val_pass_rate,_ ,_ = self._save_values(batch_i)
            test_correct: int = int(test_pass_rate * len(self.testing_labels))
            validation_correct: int = int(val_pass_rate * len(self.validation_labels))
            val_pass_rate = validation_correct / len(self.validation_labels)

            val_str = f"{'Validation':10} | {f'{validation_correct:,} / {len(self.validation_labels):,}':15} | {f'{val_pass_rate:.2%}':8} | {f'{((batch_i+1) * self.batch_size):,}':12} | {f'{(batch_i+1):,}':12} | {f'{(epoch_i+1):,}':6}"
            print(val_str + f" || {'Testing':6} | {f'{test_correct:,} / {len(self.testing_labels):,}':15} | {f'{test_pass_rate:.2%}':8}")


    # =============================== Logging ===================================
    
    def _save_values(self, batch_i: int):
        """Evalutes the network on the validation and testing set and saves them to memory, used for plotting"""
        val_pass_rate = self._try_validation() / len(self.validation_labels)
        val_cost = self.cost(self.validation_labels[-1], self.z[-1], self.a[-1])
        self.val_cost_list.append(val_cost)
        self.val_pass_rate_list.append(val_pass_rate)
        
        test_pass_rate = self._try_testing() / len(self.testing_labels)
        test_cost = self.cost(self.testing_labels[-1], self.z[-1], self.a[-1])
        self.test_cost_list.append(test_cost)
        self.test_pass_rate_list.append(test_pass_rate)

        self.examples_list.append((batch_i+1) * self.batch_size)  # adds total examples done so far    

        return test_pass_rate, val_pass_rate, test_cost, val_cost


    # =============================== Plotting ===================================
   
    def show_plot(self) -> None:
        """Plots the pass rate over time and cost over time for both the validation and testing set"""
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(self.examples_list, self.val_pass_rate_list, label="Validation")
        ax1.plot(self.examples_list, self.test_pass_rate_list, label="Testing")
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.set_xlabel("Examples")
        ax1.set_ylabel("Pass Rate (%)")
        ax1.set_title("Pass Rate over time")

        ax2.plot(self.examples_list, self.val_cost_list, label="Validation")
        ax2.plot(self.examples_list, self.test_cost_list, label="Testing")
        ax2.legend()
        ax2.set_xlabel("Examples")
        ax2.set_ylabel("Cost")
        ax2.set_title("Cost over time")

        plt.tight_layout()
        plt.show()


    # =============================== Try Data Sets ===================================
    
    def _try_validation(self) -> int:
        return self._try_set(self.validation_set, self.validation_labels)

    def _try_testing(self) -> int:
        return self._try_set(self.testing_set, self.testing_labels)

    def _try_set(self, data_set, labels) -> int:
        """Evalutes the network on a specific data set, returns the amount of correct answers"""
        # TODO add cost averaging and return, add flag if we wanna do it
        # feedforward the validation set
        amount_correct: int = 0
        for input_layer, label in zip(data_set, labels):  # type: ignore
            self._feedforward(input_layer)
            if(self._correct_answer(label)): 
                amount_correct += 1
        return amount_correct

    def _correct_answer(self, label) -> bool:
        """Checks the output layer of the network, whichever neuron has the highest activation is
        chosen as the class that is the networks output. At ties, the lowest index is chosen for simplicity."""
        correct_index = self.class_dict[label]
        output_index = self.a[-1].argmax()
        if(correct_index == output_index):
            return True
        return False


    # =============================== Setters ===================================
    
    def set_hidden_layers(self, *args: list[int]) -> None:
        self.hidden_layer_sizes = args[0]

    def set_learning_rate(self, learning_rate: float, mode: str, factor: int=128) -> None:
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate

        if mode == "fixed":
            self.learning_rate_mode = "fixed"
        elif mode == "variable":
            self.learning_rate_mode = "variable"
            self.learning_rate_factor = factor

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def set_classes(self, *args) -> None:
        self.output_layer_size = len(args)  # output layer is as big as the amount of classes
        # creates a dictionary with all classes so a desired output can be constructed from labels
        index_list: list = [x for x in range(len(args))]

        class_dict = {}
        for value, index in zip(args, index_list):  # type: ignore
            class_dict[value] = index
        self.class_dict = class_dict

    def set_logistic(self, logistic_class):
        self.logistic = logistic_class.logistic
        self.d_logistic = logistic_class.d_logistic

    def set_cost(self, cost_class):
        self.cost = cost_class.cost
        self.delta = cost_class.delta

    def set_regularization(self, reg: float):
        self.regularization = reg
  
    def set_no_improvement_offset(self, offset: int):
        self.no_improvement_offset = offset


    # =============================== Data Loading ===================================
    
    def load_data(self, training_inputs, training_labels, validation_inputs, validation_labels, test_inputs, test_labels):
        self.training_set = training_inputs
        self.training_labels = training_labels

        self.validation_set = validation_inputs
        self.validation_labels = validation_labels

        self.testing_set = test_inputs
        self.testing_labels = test_labels

        # All input layers sizes have to be same, so the size of any example is the size of the input layer
        self.input_layer_size = len(training_inputs[0])
        