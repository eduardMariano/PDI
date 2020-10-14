import Algorithms.CNN as cnn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main_mlp():
    return None

def main_cnn():
    CNN = cnn.CNN(64, 20, 10)
    train_ds = CNN.load_dataset("https://drive.google.com/file/d/1gqHi76hvQkoicJA11o9F6tqQmOep-1kA/view?usp=sharing", "training")

    print(train_ds)

    # (train_X, train_Y), (test_X, test_Y) = CNN.load_data()
    # train_X, test_X = CNN.reshape(train_X, test_X)
    # train_X, test_X = CNN.formatData(train_X, test_X)
    # train_Y_one_hot, test_Y_one_hot = CNN.load_one_hot(train_Y, test_Y)
    # train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
    # CNN.init_model()
    # CNN.compile()
    # CNN.train(train_X, train_label, valid_X, valid_label)
    # CNN.test(test_X, test_Y_one_hot)
    #
    # predicted_classes = CNN.predict(test_X)
    # predicted_classes = CNN.format_predict(predicted_classes)
    # CNN.print_predict(predicted_classes, test_X, test_Y)
    # CNN.report(predicted_classes, test_Y)

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
    main_cnn()

if __name__ == '__main__':
    main()



