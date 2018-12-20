import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from keras.datasets import fashion_mnist

multiple_classes_data = datasets.load_wine()
multiclass_classifier = OneVsRestClassifier(LogisticRegression())

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train[0].shape)

images_train =  []
for image_train in x_train:
    images_train.append(image_train.flatten())
    
images_test = []
for image_test in x_test:
    images_test.append(image_test.flatten())

images_train = np.array(images_train)
images_test = np.array(images_test)

fashion_mnist_classifier = OneVsRestClassifier(LogisticRegression(verbose=1, max_iter=4))
fashion_mnist_classifier.fit(images_train, y_train);

conf_matrix = confusion_matrix(y_test, fashion_mnist_classifier.predict(images_test))
print("CONFUSION MATRIX:")
print(conf_matrix)

print("SCORE ONE:")
print(fashion_mnist_classifier.score(images_test, y_test))

multi_class_fashion_mnist_classifier = LogisticRegression(verbose=1, max_iter=4, multi_class="multinomial", solver="sag")
multi_class_fashion_mnist_classifier.fit(images_train, y_train)

conf_matrix = confusion_matrix(y_test, multi_class_fashion_mnist_classifier.predict(images_test))
print("CONFUSION MATRIX:")
print(conf_matrix)

print("SCORE TWO:")
print(multi_class_fashion_mnist_classifier.score(images_test, y_test))