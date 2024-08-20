import numpy as np
import h5py
import copy
import pickle

# Load MNIST data
MNIST_data = h5py.File('MNIST_data.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))
MNIST_data.close()

####################################################################################
# Implementation of stochastic gradient descent algorithm

class NN:
    first_layer = {}
    second_layer = {}

    def __init__(self, inputs, hidden, outputs):
        # Initialize the model parameters, including the first and second layer parameters and biases
        self.first_layer['para'] = np.random.randn(hidden, inputs) / np.sqrt(inputs)
        self.first_layer['bias'] = np.random.randn(hidden, 1) / np.sqrt(hidden)
        self.second_layer['para'] = np.random.randn(outputs, hidden) / np.sqrt(hidden)
        self.second_layer['bias'] = np.random.randn(outputs, 1) / np.sqrt(hidden)
        self.input_size = inputs
        self.hid_size = hidden
        self.output_size = outputs

    def __activationfunc(self, Z, type='ReLU', deri=False):
        if type == 'ReLU':
            if deri:
                return np.array([1 if i > 0 else 0 for i in np.squeeze(Z)])
            else:
                return np.array([i if i > 0 else 0 for i in np.squeeze(Z)])
        elif type == 'Sigmoid':
            if deri:
                return 1 / (1 + np.exp(-Z)) * (1 - 1 / (1 + np.exp(-Z)))
            else:
                return 1 / (1 + np.exp(-Z))
        elif type == 'tanh':
            if deri:
                return 1 - (np.tanh(Z))**2
            else:
                return np.tanh(Z)
        else:
            raise TypeError('Invalid type!')

    def __Softmax(self, z):
        return 1 / sum(np.exp(z)) * np.exp(z)

    def __cross_entropy_error(self, v, y):
        return -np.log(v[y])

    def __forward(self, x, y):
        # Implement the forward computation, calculation of prediction list and error
        Z = np.matmul(self.first_layer['para'], x).reshape((self.hid_size, 1)) + self.first_layer['bias']
        H = np.array(self.__activationfunc(Z)).reshape((self.hid_size, 1))
        U = np.matmul(self.second_layer['para'], H).reshape((self.output_size, 1)) + self.second_layer['bias']
        predict_list = np.squeeze(self.__Softmax(U))
        error = self.__cross_entropy_error(predict_list, y)
        
        dic = {
            'Z': Z,
            'H': H,
            'U': U,
            'f_X': predict_list.reshape((1, self.output_size)),
            'error': error
        }
        return dic

    def __back_propagation(self, x, y, f_result):
        # Implement the backpropagation process, compute the gradients
        E = np.array([0] * self.output_size).reshape((1, self.output_size))
        E[0][y] = 1
        dU = (-(E - f_result['f_X'])).reshape((self.output_size, 1))
        db_2 = copy.copy(dU)
        dC = np.matmul(dU, f_result['H'].transpose())
        delta = np.matmul(self.second_layer['para'].transpose(), dU)
        db_1 = delta.reshape(self.hid_size, 1) * self.__activationfunc(f_result['Z'], deri=True).reshape(self.hid_size, 1)
        dW = np.matmul(db_1.reshape((self.hid_size, 1)), x.reshape((1, self.input_size)))

        grad = {
            'dC': dC,
            'db_2': db_2,
            'db_1': db_1,
            'dW': dW
        }
        return grad

    def __optimize(self, b_result, learning_rate):
        # Update the hyperparameters
        self.second_layer['para'] -= learning_rate * b_result['dC']
        self.second_layer['bias'] -= learning_rate * b_result['db_2']
        self.first_layer['bias'] -= learning_rate * b_result['db_1']
        self.first_layer['para'] -= learning_rate * b_result['dW']

    def __loss(self, X_train, Y_train):
        # Implement the loss function of the training set
        loss = 0
        for n in range(len(X_train)):
            y = Y_train[n]
            x = X_train[n][:]
            loss += self.__forward(x, y)['error']
        return loss

    def train(self, X_train, Y_train, num_iterations=1000, learning_rate=0.5):
        # Generate a random list of indices for the training set
        rand_indices = np.random.choice(len(X_train), num_iterations, replace=True)
        
        def l_rate(base_rate, ite, num_iterations, schedule=False):
            # Determine whether to use the learning schedule
            if schedule:
                return base_rate * 10 ** (-np.floor(ite / num_iterations * 5))
            else:
                return base_rate

        count = 1
        for i in rand_indices:
            f_result = self.__forward(X_train[i], Y_train[i])
            b_result = self.__back_propagation(X_train[i], Y_train[i], f_result)
            self.__optimize(b_result, l_rate(learning_rate, i, num_iterations, True))
            
            if count % 1000 == 0:
                if count % 5000 == 0:
                    loss = self.__loss(X_train, Y_train)
                    test = self.testing(x_test, y_test)
                    print(f'Trained for {count} times, loss = {loss}, test = {test}')
                else:
                    print(f'Trained for {count} times')
            count += 1

        print('Training finished!')

    def testing(self, X_test, Y_test):
        # Test the model on the test dataset
        total_correct = 0
        for n in range(len(X_test)):
            y = Y_test[n]
            x = X_test[n][:]
            prediction = np.argmax(self.__forward(x, y)['f_X'])
            if prediction == y:
                total_correct += 1
        accuracy = total_correct / float(len(X_test))
        print(f'Accuracy Test: {accuracy}')
        return accuracy

    def save_model(self, file_name):
        # Save the model parameters to a file
        model_data = {
            'first_layer': self.first_layer,
            'second_layer': self.second_layer,
            'input_size': self.input_size,
            'hid_size': self.hid_size,
            'output_size': self.output_size
        }
        with open(file_name, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {file_name}")

    def load_model(self, file_name):
        # Load the model parameters from a file
        with open(file_name, 'rb') as f:
            model_data = pickle.load(f)
        self.first_layer = model_data['first_layer']
        self.second_layer = model_data['second_layer']
        self.input_size = model_data['input_size']
        self.hid_size = model_data['hid_size']
        self.output_size = model_data['output_size']
        print(f"Model loaded from {file_name}")

####################################################################################

# Set the number of iterations
num_iterations = 200000
# Set the base learning rate
learning_rate = 0.01
# Number of inputs
num_inputs = 28 * 28
# Number of outputs
num_outputs = 10
# Size of hidden layer
hidden_size = 300

# Data fitting, training, and accuracy evaluation
model = NN(num_inputs, hidden_size, num_outputs)
model.train(x_train, y_train, num_iterations=num_iterations, learning_rate=learning_rate)
model.save_model('trained_model.pkl')  # Save the trained model

# Load the model (if needed)
# model.load_model('trained_model.pkl')

# Test the model
accu = model.testing(x_test, y_test)
