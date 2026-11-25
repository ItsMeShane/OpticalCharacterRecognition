import numpy as np
import h5py
import pickle
from scipy import ndimage


def load_mnist(file_name='MNIST_data.hdf5'):
    with h5py.File(file_name, 'r') as f:
        x_train = np.float32(f['x_train'][:])
        y_train = np.int64(f['y_train'][:, 0])
        x_test = np.float32(f['x_test'][:])
        y_test = np.int64(f['y_test'][:, 0])
    return x_train, y_train, x_test, y_test


# Data augmentation: every time an image is used it gets a new random distortion,
# so the network learns what digits look like when rotated, slanted, resized,
# or shifted, instead of memorizing the exact training images

def augment(X, rng):
    """Randomly distort a batch of flattened 28x28 images, X of shape (batch, 784)."""
    B = len(X)
    angle = np.deg2rad(rng.uniform(-15, 15, B))  # rotation
    slant = rng.uniform(-0.3, 0.3, B)            # italic-style shear
    scale = rng.uniform(0.85, 1.15, B)           # size
    shift = rng.uniform(-2, 2, (B, 2))           # position, in pixels

    # For every output pixel, work out which point of the input image to sample
    cos, sin = np.cos(angle), np.sin(angle)
    rotate = np.stack([np.stack([cos, -sin], -1), np.stack([sin, cos], -1)], -2)
    shear = np.tile(np.eye(2), (B, 1, 1))
    shear[:, 1, 0] = slant
    M = rotate @ shear / scale[:, None, None]
    grid = np.indices((28, 28)).reshape(2, -1) - 14.0  # pixel coordinates relative to the center
    src = M @ (grid[None] - shift[:, :, None]) + 14.0   # (batch, 2, 784)

    batch_index = np.broadcast_to(np.arange(B)[:, None], (B, 784))
    out = ndimage.map_coordinates(X.reshape(B, 28, 28), [batch_index, src[:, 0], src[:, 1]], order=1)
    out = out.reshape(B, 28, 28)

    # Thicker strokes for some images (small drawings end up with thick lines)
    thick = rng.random(B) < 0.3
    if thick.any():
        out[thick] = ndimage.grey_dilation(out[thick], size=(1, 2, 2))
    return out.reshape(B, 784)


####################################################################################
# Two-layer network trained with mini-batch stochastic gradient descent

class NN:
    def __init__(self, inputs, hidden, outputs, seed=None):
        self.rng = np.random.default_rng(seed)
        # He initialization (scaled for ReLU), biases start at zero
        self.first_layer = {
            'para': self.rng.standard_normal((hidden, inputs), dtype=np.float32) * np.float32(np.sqrt(2 / inputs)),
            'bias': np.zeros((hidden, 1), np.float32),
        }
        self.second_layer = {
            'para': self.rng.standard_normal((outputs, hidden), dtype=np.float32) * np.float32(np.sqrt(1 / hidden)),
            'bias': np.zeros((outputs, 1), np.float32),
        }
        self.input_size = inputs
        self.hid_size = hidden
        self.output_size = outputs

    def __forward(self, X):
        # Processes a whole batch at once: X has shape (batch, inputs)
        Z = X @ self.first_layer['para'].T + self.first_layer['bias'].T
        H = np.maximum(Z, 0)  # ReLU
        U = H @ self.second_layer['para'].T + self.second_layer['bias'].T
        P = np.exp(U - U.max(axis=1, keepdims=True))  # softmax, shifted so exp() can't overflow
        P /= P.sum(axis=1, keepdims=True)
        return Z, H, P

    def __back_propagation(self, X, y, Z, H, P):
        # Gradients of the cross-entropy loss, averaged over the batch
        dU = P.copy()
        dU[np.arange(len(y)), y] -= 1
        dU /= len(y)
        dZ = (dU @ self.second_layer['para']) * (Z > 0)
        return {
            'dC': dU.T @ H,
            'db_2': dU.sum(axis=0)[:, None],
            'dW': dZ.T @ X,
            'db_1': dZ.sum(axis=0)[:, None],
        }

    def __optimize(self, grad, learning_rate, momentum, weight_decay):
        # SGD with momentum. Weight decay pulls the weights toward zero, which
        # discourages overfitting.
        updates = [
            (self.first_layer, 'para', grad['dW'] + weight_decay * self.first_layer['para']),
            (self.first_layer, 'bias', grad['db_1']),
            (self.second_layer, 'para', grad['dC'] + weight_decay * self.second_layer['para']),
            (self.second_layer, 'bias', grad['db_2']),
        ]
        for i, (layer, key, g) in enumerate(updates):
            self.velocity[i] = momentum * self.velocity[i] - learning_rate * g
            layer[key] += self.velocity[i]

    def train(self, X_train, Y_train, X_test, Y_test, epochs=30, batch_size=64,
              learning_rate=0.05, momentum=0.9, weight_decay=1e-4, augmentation=True):
        n = len(X_train)
        total_steps = epochs * int(np.ceil(n / batch_size))
        self.velocity = [np.zeros_like(p) for p in (
            self.first_layer['para'], self.first_layer['bias'],
            self.second_layer['para'], self.second_layer['bias'])]

        step = 0
        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(n)  # visit every image once per epoch, in a new order
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                X, y = X_train[idx], Y_train[idx]
                if augmentation:
                    X = augment(X, self.rng)
                Z, H, P = self.__forward(X)
                epoch_loss += -np.log(P[np.arange(len(y)), y] + 1e-12).sum()
                grad = self.__back_propagation(X, y, Z, H, P)
                # Cosine schedule: the learning rate glides from its starting value down to 0
                lr = float(learning_rate * 0.5 * (1 + np.cos(np.pi * step / total_steps)))
                self.__optimize(grad, lr, momentum, weight_decay)
                step += 1
            test = self.testing(X_test, Y_test, verbose=False)
            print(f'Epoch {epoch}/{epochs}: loss = {epoch_loss / n:.4f}, test accuracy = {test:.4f}')

        print('Training finished!')

    def testing(self, X_test, Y_test, verbose=True):
        # Test the model on the test dataset
        _, _, P = self.__forward(X_test)
        accuracy = float(np.mean(P.argmax(axis=1) == Y_test))
        if verbose:
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

if __name__ == '__main__':
    # Number of passes over the training set
    epochs = 30
    # Number of images per gradient step
    batch_size = 64
    # Starting learning rate (decays to 0 over training)
    learning_rate = 0.05
    # Number of inputs
    num_inputs = 28 * 28
    # Number of outputs
    num_outputs = 10
    # Size of hidden layer
    hidden_size = 300

    x_train, y_train, x_test, y_test = load_mnist()

    # Data fitting, training, and accuracy evaluation
    model = NN(num_inputs, hidden_size, num_outputs, seed=0)
    model.train(x_train, y_train, x_test, y_test, epochs=epochs, batch_size=batch_size,
                learning_rate=learning_rate)
    model.save_model('trained_model.pkl')  # Save the trained model

    # Load the model (if needed)
    # model.load_model('trained_model.pkl')

    # Test the model
    accu = model.testing(x_test, y_test)
