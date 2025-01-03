import numpy as np
import random
import os

class FNN():
    def __init__(self):
        # Initialize FNN parameters
        self.input_size = 1024
        self.num_hidden = 3
        self.hidden = [512, 256, 128]
        self.hidden_size = 150
        self.output_size = 10
        self.folder_path = os.path.join(os.path.dirname(__file__), "Weights and Biases")

        self.layers = []
        self.weights = []
        self.biases = []

        # Initialize layers, weights, and biases lists
        self.layers.append(np.zeros(self.input_size))
        for size in self.hidden:
            self.layers.append(np.zeros(size))
        self.layers.append(np.zeros(self.output_size))

        for i in range(self.num_hidden + 1): 
            weight_file = os.path.join(self.folder_path, f'weight_layer_{i}.npy')
            self.weights.append(np.load(weight_file))

        for i in range(self.num_hidden + 1):
            bias_file = os.path.join(self.folder_path, f'bias_layer_{i}.npy')
            self.biases.append(np.load(bias_file))

    def SoftMax(self, x):
        # Return SoftMax of layer
        prob = np.exp(x - np.max(x))
        return prob / prob.sum(axis=0)
    
    def ReLU(self, x):
        # Return ReLU of layer
        return np.maximum(x, 0)
    
    def FeedForward(self, x):
        # Set input layer, and initialize list of z values
        self.layers[0] = x
        zs = []

        # Feedforward calclations using linear algebra
        for i in range(self.num_hidden):
            z = np.dot(self.weights[i], self.layers[i]) + self.biases[i]
            zs.append(z)
            self.layers[i+1] = self.ReLU(z)
        z = np.dot(self.weights[-1], self.layers[-2]) + self.biases[-1]
        zs.append(z)
        self.layers[-1] = self.SoftMax(z)

        return self.layers, zs

class Training():
    def __init__(self, train_data, test_data, learning_rate = 0.01):
        self.FNN = FNN()
        self.train_data = train_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.outputs = {
            "angel":0,
            "basketball":1,
            "car":2,
            "cat":3,
            "crab":4,
            "dolphin":5,
            "helicopter":6,
            "mushroom":7,
            "octopus":8,
            "skull":9
        }

    def dReLU(self, x):
        # Return derivative of ReLU function
        return np.where(x > 0, 1, 0)
    
    def gradient_descent(self, inputs):
        # Initialize weight and bias gradients
        w_gradients = []
        b_gradients = []

        for i in range(len(inputs)):
            # Normalize / Vectorize the image, get list of activations, z values, and set up the expected output vector
            train_image = inputs[i]["image"].reshape(-1) / 255.0
            a, z = self.FNN.FeedForward(train_image)
            expected_output = np.zeros(self.FNN.output_size)
            expected_output[self.outputs[inputs[i]]["label"]] = 1

            # Call backpropogation to get gradient vectors
            w_gradient, b_gradient = self.backpropogation(a, z, expected_output)

            # Add gradient vectors to list for training batch
            w_gradients.append(w_gradient)
            b_gradients.append(b_gradient)

        # Get average gradient vectors
        w_gradient_avg = [np.mean([w[i] for w in w_gradients], axis=0) for i in range(len(w_gradients[0]))]
        b_gradient_avg = [np.mean([b[i] for b in b_gradients], axis=0) for i in range(len(b_gradients[0]))]


        # Adjust weights and biases based off gradient vectors and learning rate
        for i in range(self.FNN.num_hidden + 1):
            self.FNN.weights[i] -= w_gradient_avg[i] * self.learning_rate
            self.FNN.biases[i] -= b_gradient_avg[i] * self.learning_rate

    def backpropogation(self, a, z, e):
        # Initialize gradient vectors
        b_gradient = [np.zeros(b.shape) for b in self.FNN.biases]
        w_gradient = [np.zeros(w.shape) for w in self.FNN.weights]

        # Set output layer gradients, SoftMax with Cross-Entropy Loss
        b_gradient[-1] = a[-1] - e 
        w_gradient[-1] = np.dot(b_gradient[-1].reshape(-1, 1), a[-2].reshape(1, -1))

        # Set hidden layer gradients, ReLU
        for i in range(2, self.FNN.num_hidden + 2):
            b_gradient[-i] = np.dot(self.FNN.weights[-i + 1].T, b_gradient[-i + 1]) * self.dReLU(z[-i])
            w_gradient[-i] = np.dot(b_gradient[-i].reshape(-1, 1), a[-i - 1].reshape(1, -1))

        # Return gradient vectors
        return w_gradient, b_gradient
    
    def update_parameters(self):
        for i, w in enumerate(self.FNN.weights):
            np.save(os.path.join(self.FNN.folder_path, f'weight_layer_{i}.npy'), w)

        for i, b in enumerate(self.FNN.biases):
            np.save(os.path.join(self.FNN.folder_path, f'bias_layer_{i}.npy'), b)
    
    def train(self, num_epochs, batch_size):
        for i in range(num_epochs):
            shuffled_train_data = random.shuffle(self.train_data)

            for j in range(1, len(shuffled_train_data) // batch_size):
                index = j * batch_size
                self.gradient_descent(shuffled_train_data[index-batch_size:index])
                percent_done = float(j / (len(shuffled_train_data) // batch_size))
                print(percent_done, end="\r")

            self.learning_rate *= 0.999
            
            self.update_parameters()

            test = Test(self.test_data)
            accuracy = test.evaluate()
            print(f"Accuracy for epoch {str(i)}: {str(accuracy)}")

class Test():
    def __init__(self, test_data):
        self.FNN = FNN()
        self.test_data = test_data
        self.outputs = {
            "angel":0,
            "basketball":1,
            "car":2,
            "cat":3,
            "crab":4,
            "dolphin":5,
            "helicopter":6,
            "mushroom":7,
            "octopus":8,
            "skull":9
        }
    
    def evaluate(self):
        correct = 0

        for i in range(len(self.test_data)):
            test_image = self.test_data[i]["image"].reshape(-1) / 255.0
            activations, _ = self.FNN.FeedForward(test_image)
            if np.argmax(activations[-1]) == self.outputs[self.test_data[i]]: correct += 1

        return correct / len(self.test_data)

if __name__ == "__main__":
    Train_Data = np.load("Doodle_Data_Train.npy", allow_pickle=True).tolist()
    Test_Data = np.load("Doodle_Data_Test.npy", allow_pickle=True).tolist()
    Train = Training(Train_Data, Test_Data)
    Train.train(50, 60)