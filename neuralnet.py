import random
import math

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Weights and biases initialization
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        self.weights.append([[random.uniform(-1, 1) for _ in range(hidden_sizes[0])] for _ in range(input_size)])
        self.biases.append([random.uniform(-1, 1) for _ in range(hidden_sizes[0])])
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.weights.append([[random.uniform(-1, 1) for _ in range(hidden_sizes[i + 1])] for _ in range(hidden_sizes[i])])
            self.biases.append([random.uniform(-1, 1) for _ in range(hidden_sizes[i + 1])])
        
        # Last hidden layer to output
        self.weights.append([[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_sizes[-1])])
        self.biases.append([random.uniform(-1, 1) for _ in range(output_size)])

    def feedforward(self, inputs):
        layer_inputs = inputs
        all_layer_inputs = [inputs]  # Store inputs of all layers for backpropagation

        for i in range(len(self.hidden_sizes)):
            hidden = [0] * self.hidden_sizes[i]
            for j in range(self.hidden_sizes[i]):
                for k in range(len(layer_inputs)):
                    hidden[j] += layer_inputs[k] * self.weights[i][k][j]
                hidden[j] += self.biases[i][j]
                hidden[j] = sigmoid(hidden[j])
            layer_inputs = hidden
            all_layer_inputs.append(hidden)
        
        output = [0] * self.output_size
        for i in range(self.output_size):
            for j in range(self.hidden_sizes[-1]):
                output[i] += layer_inputs[j] * self.weights[-1][j][i]
            output[i] += self.biases[-1][i]
            output[i] = sigmoid(output[i])
        
        all_layer_inputs.append(output)

        return output, all_layer_inputs

    def train(self, inputs, targets, learning_rate=0.1):
        _, all_layer_inputs = self.feedforward(inputs)
        
        # Backpropagation
        output_errors = [targets[i] - all_layer_inputs[-1][i] for i in range(self.output_size)]

        for i in range(self.output_size):
            for j in range(self.hidden_sizes[-1]):
                self.weights[-1][j][i] += learning_rate * output_errors[i] * sigmoid_derivative(all_layer_inputs[-1][i]) * all_layer_inputs[-2][j]
            self.biases[-1][i] += learning_rate * output_errors[i] * sigmoid_derivative(all_layer_inputs[-1][i])

        errors = [output_errors]
        for i in reversed(range(len(self.hidden_sizes))):
            layer_errors = [0] * self.hidden_sizes[i]
            for j in range(self.hidden_sizes[i]):
                error = 0
                for k in range(len(errors[0])):
                    error += errors[0][k] * self.weights[i + 1][j][k]
                layer_errors[j] = error
            errors.insert(0, layer_errors)

        for i in reversed(range(len(self.hidden_sizes))):
            for j in range(self.hidden_sizes[i]):
                for k in range(len(all_layer_inputs[i])):
                    self.weights[i][k][j] += learning_rate * errors[0][j] * sigmoid_derivative(all_layer_inputs[i + 1][j]) * all_layer_inputs[i][k]
                self.biases[i][j] += learning_rate * errors[0][j] * sigmoid_derivative(all_layer_inputs[i + 1][j])
            errors.pop(0)

def get_input_list(prompt):
    return list(map(int, input(prompt).split()))

def get_training_data():
    training_data = []
    num_samples = int(input("Enter the number of samples: "))
    for _ in range(num_samples):
        inputs = get_input_list("Enter the input values separated by spaces: ")
        targets = get_input_list("Enter the target values separated by spaces: ")
        training_data.append((inputs, targets))
    return training_data

# Define the problem
choice = input("Do you want to use existing training data or create new ones? (e/n): ").strip().lower()

if choice == "n":
    training_data = get_training_data()
else:
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]

# Initialize the neural network
nn = NeuralNetwork(2, [2], 1)  # 2 input nodes, 2 hidden nodes, 1 output node

# Train the network
for epoch in range(500000):  # Number of epochs
    for inputs, targets in training_data:
        nn.train(inputs, targets)

# Measure error
def mean_squared_error(targets, outputs):
    return sum((t - o) ** 2 for t, o in zip(targets, outputs)) / len(targets)


# Train the network and monitor error
for epoch in range(5000000):  # Number of epochs
    total_error = 0
    for inputs, targets in training_data:
        nn.train(inputs, targets)
        outputs = nn.feedforward(inputs)[0]
        total_error += mean_squared_error(targets, outputs)
    
    if epoch % 1000 == 0:  # Print error every 1000 epochs
        print(f"Epoch {epoch}, Error: {total_error / len(training_data)}")

# Test the network
for inputs, _ in training_data:
    print(f"Input: {inputs}, Output: {nn.feedforward(inputs)[0]}")