from keras.datasets import fashion_mnist
from tensorflow.python.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import to_categorical
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

class Multilayer_Perceptron:
    def __init__(self, layers_size, epochs, batch_size, activation, neurons):
        self.hidden_layer = layers_size
        self.model = []
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = 0
        self.activation = activation
        self.num_neurons = neurons
        self.dim_data = []

    def load_data(self):
        return fashion_mnist.load_data()

    def set_num_classes(self, train):
        self.classes = np.unique(train)
        self.num_classes = len(self.classes)

    def configure(self, train_X, test_X):
        (train_X, test_X) = self.reshape(train_X, test_X)
        return self.formatData(train_X, test_X)

    def reshape(self, train, test):
        self.dim_data = np.prod(train.shape[1:])
        train = train.reshape(train.shape[0], self.dim_data)
        test = test.reshape(train.shape[0], self.dim_data)
        return train, test

    def formatData(self, train, test):
        train = train.astype('float32') / 255.0
        test = test.astype('float32') / 255.0
        return (train, test)

    def init_model(self):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=self.dim_data))
        for i in self.hidden_layer:
            self.model.add(Dense(self.num_neurons, activation=self.activation))
            self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

    def compile(self):
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
        return self.model

    def train(self, train_X, train_Y, test_X, test_Y):
        return self.model.fit(train_X, to_categorical(train_Y), batch_size=self.batch_size,epochs=self.epochs,verbose=1,validation_data=(test_X, to_categorical(test_Y)))

    def test(self, test_X, test_Y):
        return self.model.evaluate(test_X, to_categorical(test_Y))

    def predict(self, test_X):
        return self.model.predict(test_X)

    def format_predict(self, predicted_classes):
        return np.argmax(np.round(predicted_classes), axis=1)

    def print_predict(self, predicted_classes, test_X, test_Y):
        correct = np.where(predicted_classes == test_Y)[0]
        print("Found %d correct labels" % len(correct))
        for i, correct in enumerate(correct[:9]):
            plt.subplot(3, 3, i + 1)
            plt.imshow(test_X[correct].reshape(28, 28), cmap='gray', interpolation='none')
            plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
            plt.tight_layout()

    def report(self, predicted_classes, test_Y):
        target_names = ["Class {}".format(i) for i in range(self.num_classes)]
        print(classification_report(test_Y, predicted_classes, target_names=target_names))