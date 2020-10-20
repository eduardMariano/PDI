import Algorithms.CNN as cnn
import Algorithms.MLP as mlp

def main_mlp():
    MLP = mlp.Multilayer_Perceptron(1, 10, 64, "linear", 40)
    (train_X, train_Y), (test_X, test_Y) = MLP.load_data()
    MLP.set_num_classes(train_Y)
    (train_X, test_X, dim_data) = MLP.configure(train_X, test_X)
    # train_X, valid_X, train_label, valid_label = MLP.split(train_X, train_Y)

    MLP.init_model()
    MLP.compile()
    train = MLP.train(train_X, train_Y, test_X, test_Y)
    test_eval = MLP.test(test_X, test_Y)

    MLP.plot(train)

    predicted_classes = MLP.predict(test_X)
    predicted_classes = MLP.format_predict(predicted_classes)
    MLP.print_predict(predicted_classes, test_X, test_Y)
    MLP.report(predicted_classes, test_Y)

def main_cnn():
    CNN = cnn.CNN(1, 10, 64, "linear", 40)
    (train_X, train_Y), (test_X, test_Y) = CNN.load_data()
    CNN.set_num_classes(train_Y)
    (train_X, test_X) = CNN.configure(train_X, test_X)
    train_X, valid_X, train_label, valid_label = CNN.split(train_X, train_Y)

    CNN.init_model()
    CNN.compile()
    train = CNN.train(train_X, valid_X, train_label, valid_label)
    test_eval = CNN.test(test_X, test_Y)

    CNN.plot(train)

    predicted_classes = CNN.predict(test_X)
    predicted_classes = CNN.format_predict(predicted_classes)
    CNN.print_predict(predicted_classes, test_X, test_Y)
    CNN.report(predicted_classes, test_Y)

def main():
    ""
    # img = imread(in_filename)
    #
    # if which_axis == 'r':
    #     out = crop_r(img, scale)
    # elif which_axis == 'c':
    #     out = crop_c(img, scale)
    # else:
    #     print('usage: carver.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
    #     sys.exit(1)
    #
    # imwrite(out_filename, out)
    main_mlp()

if __name__ == '__main__':
    main()



