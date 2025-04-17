from mnist.loader import MNIST
import numpy as np
from tqdm import tqdm

mndata = MNIST("archive")
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

training_data = list(zip(train_labels, train_images))
testing_data = list(zip(test_labels, test_images))

def create_batches(data, batch_size):
    np.random.shuffle(data)
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    return batches

class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(10, 784) * np.sqrt(2/784)
        self.b1 = np.zeros((10, 1))
        self.W2 = np.random.randn(10, 10) * np.sqrt(2/10)
        self.b2 = np.zeros((10, 1))
        self.W3 = np.random.randn(10, 10) * np.sqrt(2/10)
        self.b3 = np.zeros((10, 1))
        self.lr = 0.001

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def stable_softmax(self, z):
        z_shift = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_shift)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward_propagation(self, X):
        self.X = X
        self.z1 = self.W1 @ X + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = self.sigmoid(self.z2)
        self.z3 = self.W3 @ self.a2 + self.b3
        return self.z3

    def backpropagation(self, Y):
        m = Y.shape[1]
        y_hat = self.stable_softmax(self.z3)
        dz3 = (y_hat - Y) / m
        dW3 = dz3 @ self.a2.T
        db3 = np.sum(dz3, axis=1, keepdims=True)

        da2 = self.W3.T @ dz3
        dz2 = da2 * self.sigmoid_derivative(self.a2)
        dW2 = dz2 @ self.a1.T
        db2 = np.sum(dz2, axis=1, keepdims=True)

        da1 = self.W2.T @ dz2
        dz1 = da1 * self.sigmoid_derivative(self.a1)
        dW1 = dz1 @ self.X.T
        db1 = np.sum(dz1, axis=1, keepdims=True)

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, image_pixels):
        X = np.array(image_pixels, dtype=float).reshape(784, 1) / 255.0
        z3 = self.forward_propagation(X)
        probs = self.stable_softmax(z3)
        return int(np.argmax(probs))

if __name__ == "__main__":
    nn = NeuralNetwork()
    epochs = 1000
    batch_size = 32

    for epoch in range(1, epochs + 1):
        batches = create_batches(training_data, batch_size)
        with tqdm(batches, desc=f"Epoch {epoch}") as t:
            for batch in t:
                labels, images = zip(*batch)
                X = np.array(images, dtype=float).T / 255.0
                Y = np.zeros((10, len(labels)))
                for i, label in enumerate(labels):
                    Y[label, i] = 1

                nn.forward_propagation(X)
                nn.backpropagation(Y)

        # test
        correct = 0
        for label, image in testing_data:
            pred = nn.predict(image)
            correct += pred == label
        accuracy = correct / len(testing_data) * 100
        print(f"Epoch {epoch} - Test Accuracy: {accuracy:.2f}%")
