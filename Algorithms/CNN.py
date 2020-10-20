import numpy as np
from keras.datasets import fashion_mnist
from tensorflow.python.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class CNN:
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

    def split(self, train_X, train_Y):
        return train_test_split(train_X, to_categorical(train_Y), test_size=0.2, random_state=13)

    def reshape(self, train, test):
        self.dim_data = np.prod(train.shape[1:])
        train = train.reshape(train.shape[0], self.dim_data)
        test = test.reshape(train.shape[0], self.dim_data)
        return train, test

    def formatData(self, train, test):
        train = train.astype('float32')
        test = test.astype('float32')
        return (train / 255), (test / 255)

    def init_model(self, data_dim):
        self.model = Sequential()
        for i in self.hidden_layer:
            if i == 0:
                self.model.add(Conv2D(self.num_neurons, kernel_size=(3, 3), activation=self.activation, padding='same', input_shape=self.data_dim))
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