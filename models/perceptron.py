import numpy as np

class Perceptron:
    def __init__(self, input_dim, epochs=1000, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(input_dim+1)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation
    
    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

    def test(self, test_inputs, test_labels):
        predictions = []

        for inputs, label in zip(test_inputs, test_labels):
            prediction = self.predict(inputs)
            predictions.append(prediction==label)

        return np.mean(predictions)
    
    def save(self, filename):
        np.savez(filename, weights=self.weights)

    
    def load(self, filename):
        data = np.load(filename)
        self.weights = data['weights']



            
    

