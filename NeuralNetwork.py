import numpy as np
import pickle

class NN:
    def __init__(self, inputs, hidden, outputs):
        self.first_layer = {}
        self.second_layer = {}
        self.input_size = inputs
        self.hid_size = hidden
        self.output_size = outputs

    def __activfunc(self, Z, type='ReLU', deri=False):
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

    def __forward(self, x):
        Z = np.matmul(self.first_layer['para'], x).reshape((self.hid_size, 1)) + self.first_layer['bias']
        H = np.array(self.__activfunc(Z)).reshape((self.hid_size, 1))
        U = np.matmul(self.second_layer['para'], H).reshape((self.output_size, 1)) + self.second_layer['bias']
        predict_list = np.squeeze(self.__Softmax(U))
        return predict_list

    def load_model(self, file_name):
        with open(file_name, 'rb') as f:
            model_data = pickle.load(f)
        self.first_layer = model_data['first_layer']
        self.second_layer = model_data['second_layer']
        self.input_size = model_data['input_size']
        self.hid_size = model_data['hid_size']
        self.output_size = model_data['output_size']

    def predict(self, image):
        image_vector = image.flatten().reshape((784, 1))
        prediction_probs = self.__forward(image_vector)
        prediction = np.argmax(prediction_probs)
        # certainty = prediction_probs[prediction]  
        # print(f"Prediction: {prediction}, Certainty: {certainty:.4f}") 
        return prediction

