import numpy as np
import tensorflow
from keras.datasets import fashion_mnist
from tensorflow.python.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.utils import get_file
from keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
import pathlib

class CNN:
    def __init__(self, batch, epochs, num_classes):
        self.model = Sequential()
        self.batch_size = batch
        self.epochs = epochs
        self.num_classes = num_classes

    def load_data(self):
        return fashion_mnist.load_data()

    def load_dataset(self, dataset, type):
        data_dir = get_file(origin=dataset, fname='', untar=True)
        data_dir = pathlib.Path(data_dir)
        return image_dataset_from_directory(data_dir,
          validation_split=0.2,
          subset=type,
          seed=123,
          image_size=(180, 180),
          batch_size=self.batch_size)

    def set_num_classes(self, train_ds):
        class_names = train_ds.class_names
        return len(class_names)

    def labels(self, train_Y):
        cls = np.unique(train_Y)
        return cls, len(cls)

    def reshape(self, train, test):
        train = train.reshape(-1, 28, 28, 1)
        test = test.reshape(-1, 28, 28, 1)
        return train, test

    def formatData(self, train, test):
        train = train.astype('float32')
        test = test.astype('float32')
        return (train / 255), (test / 255)

    def load_one_hot(self, train_Y, test_Y):
        return to_categorical(train_Y), to_categorical(test_Y)

    def init_model(self):
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28, 28, 1), padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D((2, 2), padding='same'))
        self.model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='linear'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        return self.model

    def compile(self):
        self.model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
        return self.model

    def train(self, train_X, train_label, valid_X, valid_label):
        return self.model.fit(train_X, train_label, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                                          validation_data=(valid_X, valid_label))

    def test(self, test_X, test_Y_one_hot):
        return self.model.evaluate(test_X, test_Y_one_hot, verbose=0)

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

    def plot(self, train):
        accuracy = train.history['acc']
        val_accuracy = train.history['val_acc']
        loss = train.history['loss']
        val_loss = train.history['val_loss']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
