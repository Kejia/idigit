
import matplotlib.pyplot as plot
import numpy
import os.path
import pandas
import pickle
import sklearn.externals as external
import sklearn.svm as svm

PICKLE_SVM = 'digit_recognization_svm.pickle'
CLFSVM = None
data = pandas.read_csv('./data/train.csv')
test = pandas.read_csv('./data/test.csv')
#print('%s row, %s column\n' % data.shape)
#print(data.head())
def show(i, data):
    """
    show the image represented by the ith row.
    param:
        i: integer
        data: data from train.csv
    return:
        none.
    """
    width = height = pow(data.shape[1] - 1, 0.5)
    image = data.iloc[i, 1 : ].reshape(width, height)
    plot.imshow(image)
    plot.show()
# show the first 8 pics
##for i in range(8): show(i, data)
def scale(data):
    """
    scale pixel values to fit SVM: [0, 255] → [0, 1].
    param:
        data: the train matrix
    return:
        scaled train data.
    """
    r = None
    r = numpy.divide(data.iloc[ : , 1 : ], 255.0)
    r = numpy.concatenate((data.iloc[ : , : 1], r), axis = 1)
    return r
def train_svm():
    """
    this function trains a SVM model with default the panelty and kernel.
    param:
        None
    return:
        a SVM classifier.
    """
    classifier = svm.SVC()
    X = data.values[ : , 1 : ]
    y = data.values[ : , 0]
    classifier.fit(X, y)
    CLFSVM = classifier
    joblib.dump(classifier, PICKLE_SVM)
    return classifier
def create_svm_classifier():
    """
    create a SVM classifier.
    param:
        None
    return:
        a SVM classifier.
    """
    r = joblib.load(PICKLE_SVM) if os.path.isfile(PICKLE_SVM) else train_svm()
    CLFSVM = r
    return r
def recognize_svm(image):
    """
    recognize a digit image by using the trained SVM model.
    param:
        image: a 28 * 28 matrix representing a pixels pic, each element of the matrix is an integer in [0, 255]
    return:
        the digit of the image.
    """
    return CLFSVM.predict(image).values[0, 0] if CLFSVM else create_svm_classifier()
