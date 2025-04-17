from mnist.loader import MNIST
import numpy as np
import math
from tqdm import tqdm

mndata = MNIST("archive")
training_images, training_labels = mndata.load_training()
training_data = list(zip(training_labels, training_images))
testing_images, testing_labels = mndata.load_testing()
testing_data = list(zip(testing_labels, testing_images))

class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(10, 784) * 0.01
        self.b1 = np.zeros((10, 1), dtype=float)
        self.W2 = np.random.randn(10, 10) * 0.01
        self.b2 = np.zeros((10, 1), dtype=float)
        self.W3 = np.random.randn(10, 10) * 0.01
        self.b3 = np.zeros((10, 1), dtype=float)
        self.learning_rate = 0.001

    def sigmoid(self, a): #np.exp(2) = e**2
        return 1/(1 + np.exp(-a))
    
    def softmax(self, z):
        e_exp = ([math.e**(zi) for zi in z])
        S = sum(e_exp)
        probability_dist = [ ezi/S for ezi in e_exp]
        return probability_dist
    
    def stable_softmax(self, z):
        z = z - np.max(z)  # for numerical stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)
    
    def forward_propagation(self, image_pixels):
        if len(image_pixels) != 784:
            print("wrong image size!")
            return 
        
        self.x = np.array([[val] for val in image_pixels])
        #print(self.x.shape) #(1, 784)
        self.z1 = self.W1 @ self.x + self.b1
        #print(self.z1.shape) #(10, 1)
        self.a1 = self.sigmoid(self.z1)
        #print(self.a1.shape)#(10, 1)
        self.z2 = self.W2 @ self.a1 + self.b2
        #print(self.z2.shape)#(10, 1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = self.W3 @ self.a2 + self.b3
        #print(self.z3.shape) #(10, 1)
        return self.z3
    
    def label_to_prediction_dist(self, correct_label):
        one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        one_hot[correct_label] = 1
        y = np.array([[yi] for yi in one_hot])
        return y

    def backpropagation(self, correct_label):
        y_hat = self.stable_softmax(self.z3)
        y = self.label_to_prediction_dist(correct_label)
        dL_dz3 = y_hat - y
        delta3 = dL_dz3
        dL_db3 = delta3
        dL_dW3 = delta3@self.a2.T
        da2_dz2 = self.a2 * (1 - self.a2)
        #print(delta3.shape, self.W3.shape, da2_dz2.shape) #(10, 1) (10, 10) (10, 1)
        dL_dz2 = (self.W3.T@delta3)*da2_dz2
        delta2 = dL_dz2
        #print(delta2.shape) #(10, 1)
        dL_dW2 = delta2@self.a1.T
        dL_db2 = delta2
        da1_dz1 = self.a1 * (1-self.a1)
        delta1 = (self.W2.T@delta2) * da1_dz1
        dL_db1 = delta1
        dL_dW1 = delta1@self.x.T

        self.b3 -= (self.learning_rate * dL_db3)
        self.W3 -= (self.learning_rate * dL_dW3)
        self.b2 -= (self.learning_rate * dL_db2)
        self.W2 -= (self.learning_rate * dL_dW2)
        self.b1 -= (self.learning_rate * dL_db1)
        self.W1 -= (self.learning_rate * dL_dW1)
    
    def predict(self, image_pixels):
        self.z3 = self.forward_propagation(image_pixels)
        y_hat = self.stable_softmax(self.z3)
        prediction_distribution = [l[0] for l in y_hat.tolist()]
        prediction = prediction_distribution.index(max(prediction_distribution))
        return prediction

def test():
    nn = NeuralNetwork()
    test_image, test_label = training_images[0], training_labels[0]
    nn.forward_propagation(test_image)
    nn.backpropagation(test_label)
    #print(nn.predict(test_image))

if __name__ == "__main__":
    #test()
    nn = NeuralNetwork()

    for epoch in range(1, 11):
        with tqdm(desc=f"training {epoch}", total=len(training_data)) as pbar:
            for label, image_pixels in training_data:
                nn.forward_propagation(image_pixels)
                nn.backpropagation(label)
                pbar.update(1)

        print("testing...")
        correct, incorrect = 0, 0
        for correct_label, image_pixels in testing_data:
            predicted_label = nn.predict(image_pixels)
            if predicted_label == correct_label:
                correct += 1
            else:
                incorrect += 1
        percentage_correct = round((correct/(correct+incorrect))*100)
        print(correct, incorrect)
        print("precentage correct:", percentage_correct, "%")