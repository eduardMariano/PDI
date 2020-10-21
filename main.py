import Algorithms.CNN as cnn
import Algorithms.MLP as mlp
import Algorithms.SeamCarving as sc
from imageio import imread, imwrite
import sys

def main_mlp():
    MLP = mlp.Multilayer_Perceptron(1, 10, 64, "linear", 34, 32)
    (train_X, train_Y), (test_X, test_Y) = MLP.load_data(1)
    MLP.set_num_classes(train_Y)
    (train_X, test_X) = MLP.configure(train_X, test_X)

    MLP.init_model()
    MLP.compile()

    train = MLP.train(train_X, train_Y, test_X, test_Y)
    test_eval = MLP.test(test_X, test_Y)
    predicted_classes = MLP.predict(test_X)

    MLP.plot(train)

    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])
    MLP.report(predicted_classes, test_Y)

def main_cnn():
    CNN = cnn.CNN(1, 10, 64, "linear", 40, 28, 1)
    (train_X, train_Y), (test_X, test_Y) = CNN.load_data(1)
    CNN.set_num_classes(train_Y)
    (train_X, test_X) = CNN.configure(train_X, test_X)
    train_X, valid_X, train_label, valid_label = CNN.split(train_X, train_Y)

    CNN.init_model()
    CNN.compile()

    train = CNN.train(train_X, valid_X, train_label, valid_label)
    test_eval = CNN.test(test_X, test_Y)

    CNN.plot(train)
    predicted_classes = CNN.predict(test_X)

    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])
    CNN.report(predicted_classes, test_Y)

def main():
    # main_mlp()
    main_cnn()

if __name__ == '__main__':
    main()



