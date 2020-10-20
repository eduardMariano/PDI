from keras.datasets import fashion_mnist, cifar10
from tensorflow.python.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import to_categorical
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

class Multilayer_Perceptron:
    def __init__(self, layers_size, epochs, batch_size, activation, neurons, data_size):
        self.hidden_layer = layers_size
        self.model = []
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = 0
        self.activation = activation
        self.num_neurons = neurons
        self.data_dim = [],
        self.data_size = data_size

    def load_data(self):
        return fashion_mnist.load_data()

    def set_num_classes(self, train):
        self.classes = np.unique(train)
        self.num_classes = len(self.classes)

    def configure(self, train_X, test_X):
        (train_X, test_X) = self.reshape(train_X, test_X)
        return self.formatData(train_X, test_X)

    def reshape(self, train, test):
        self.data_dim = np.prod(train.shape[1:])
        train = train.reshape(train.shape[0], self.data_dim)
        test = test.reshape(test.shape[0], self.data_dim)
        return train, test

    def formatData(self, train, test):
        train = train.astype('float32') / 255.0
        test = test.astype('float32') / 255.0
        return (train, test)

    def init_model(self):
        self.model = Sequential()
        for i in range(self.hidden_layer):
            if i == 0:
                self.model.add(Dense(self.num_neurons, activation=self.activation, input_shape=(self.data_dim, )))
            elif i > 0:
                self.model.add(Dense(self.num_neurons, activation=self.activation))
            self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

    def compile(self):
        self.model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
        return self.model

    def train(self, train_X, train_Y, test_X, test_Y):
        return self.model.fit(train_X, to_categorical(train_Y), batch_size=self.batch_size,epochs=self.epochs,verbose=1,validation_data=(test_X, to_categorical(test_Y)))

    def test(self, test_X, test_Y):
        return self.model.evaluate(test_X, to_categorical(test_Y))

    def predict(self, test_X):
        return self.model.predict(test_X)

    def format_predict(self, predicted_classes):
        return np.argmax(np.round(predicted_classes), axis=1)

    def report(self, predicted_classes, test_Y):
        target_names = ["Class {}".format(i) for i in range(self.num_classes)]
        print(classification_report(test_Y, predicted_classes, target_names=target_names))

    def plot(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(self.epochs)

        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()