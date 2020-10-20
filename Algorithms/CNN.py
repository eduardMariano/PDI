import numpy as np
from keras.datasets import fashion_mnist, cifar10, cifar100
from tensorflow.python.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class CNN:
    def __init__(self, layers_size, epochs, batch_size, activation, neurons, data_dim, data_size):
        self.hidden_layer = layers_size
        self.model = []
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = 0
        self.activation = activation
        self.num_neurons = neurons
        self.data_dim = data_dim
        self.data_size = data_size

    def load_data(self, type):
        if type == 1:
            return fashion_mnist.load_data()
        elif type == 2:
            return cifar10.load_data()
        elif type == 3:
            return cifar100.load_data()

    def set_num_classes(self, train):
        self.classes = np.unique(train)
        self.num_classes = len(self.classes)

    def configure(self, train_X, test_X):
        (train_X, test_X) = self.reshape(train_X, test_X)
        return self.formatData(train_X, test_X)

    def split(self, train_X, train_Y):
        return train_test_split(train_X, to_categorical(train_Y), test_size=0.2, random_state=13)

    def reshape(self, train, test):
        train = train.reshape(-1, self.data_dim, self.data_dim, self.data_size)
        test = test.reshape(-1, self.data_dim, self.data_dim, self.data_size)
        return train, test

    def formatData(self, train, test):
        train = train.astype('float32')
        test = test.astype('float32')
        return (train / 255), (test / 255)

    def init_model(self):
        self.model = Sequential()
        for i in range(self.hidden_layer):
            if i == 0:
                self.model.add(Conv2D(self.num_neurons, kernel_size=(3, 3), activation=self.activation, padding='same', input_shape=(self.data_dim, self.data_dim, self.data_size)))
            elif i > 0:
                self.model.add(Conv2D(self.num_neurons, kernel_size=(3, 3), activation=self.activation, padding='same'))
            self.model.add(LeakyReLU(alpha=0.1))
            self.model.add(MaxPooling2D((2, 2), padding='same'))
            self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(self.num_neurons, activation=self.activation))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        return self.model

    def compile(self):
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        return self.model

    def train(self, train_X, valid_X, train_label, valid_label):
        return self.model.fit(train_X, train_label, batch_size=self.batch_size,epochs=self.epochs,verbose=1,validation_data=(valid_X, valid_label))

    def test(self, test_X, test_Y):
        return self.model.evaluate(test_X, to_categorical(test_Y), verbose=1)

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