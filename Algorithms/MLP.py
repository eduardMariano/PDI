from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd

class Mlp:
    def __init__(self, layers_size, iterations):
        self.hidden_layer = layers_size
        self.iterations = iterations
        self.classifier = []

    def train(self, train_X, train_Y):
        self.classifier - MLPClassifier(hidden_layer_sizes=self.hidden_layer,  max_iter=self.iterations,activation = 'relu',solver='adam',random_state=1)
        self.classifier.fit(train_X, train_Y)

    def test(self, test):
        return self.classifier.predict(test)

    def cm(self, y_pred, y_val):
        return confusion_matrix(y_pred, y_val)

    def accuracy(confusion_matrix):
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements