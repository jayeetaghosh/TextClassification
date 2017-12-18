"""
Logistic regression for Part Impact model - using Keras
This is multilevel linear model that uses BOW features + other features
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from PartImpactBOW import * #read_data, dummyEncode, merge_textcols
from PartImpactMetrics import *


FLAGS = None
MAX_BOW_FEA = None#500##570#0
MAX_STEPS = 1000
num_row = 10000  # num of rows to read, give a huge number to read all rows
# Set verbosity to INFO for more detailed log output
tf.logging.set_verbosity(tf.logging.INFO)


def main():

    dataset = read_data()
    dataset = dummyEncode(dataset)
    dataset = merge_textcols(dataset)

    bow_df = get_bow_features(dataset, num_row, MAX_BOW_FEA, keepNum=True, handleAbbr=True, ngram =[1,2])
    finalDF = get_all_features(dataset, bow_df)
    finalDF.drop('CDG_PART_NO', axis=1, inplace=True)

    print(finalDF['MODEL_NO'].value_counts())
    X = finalDF.values.copy()
    X_All, labels = X[:, :-1], X[:, -1]

    # One Hot encoding of target
    Y_All, encoder = preprocess_labels(labels)

    nb_classes = Y_All.shape[1]-1 # Missing is not a valid category
    print(nb_classes, 'classes')

    dims = X_All.shape[1]
    print(dims, 'features')

    print(np.unique(labels))

    # use keras
    # Y_trainK = keras.utils.to_categorical(labels)#, num_classes=11)
    train_size = int(finalDF.shape[0]*0.8)
    X_train = X_All[:train_size, :]
    Y_train = Y_All[:train_size,:nb_classes]
    X_test = X_All[train_size:,:]
    Y_test = Y_All[train_size:,:nb_classes]


    print("Building model...")
    model = Sequential()
    # model.add(Dense(nb_classes, input_shape=(dims,), activation='sigmoid'))
    # model.add(Activation('softmax'))
    input_dim = dims# MAX_BOW_FEA + 3
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    # model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics='accuracy')
    # model.fit(X_train, Y_train, epochs=10, batch_size=32)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', fmeasure, precision, recall])
    #
    from keras.callbacks import TensorBoard
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./LogFiles', histogram_freq=1, write_graph=True, write_images=False)
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=128, callbacks = [tbCallBack])

    # model.fit(X_train, Y_train, epochs=20, batch_size=128)
    score = model.evaluate(X_test, Y_test, batch_size=32)  # it gives loss and metrics
    print("\nFinal Validation score: categorical_crossentropy, accuracy, fmeasure, precision, recall\n", score)

    print("Really Done!")



if __name__ == '__main__':
    main()
"""
Features: 5700 binary BOW (text), 7 (structured)
Model: multilabel multilayer perceptron
Multilabel logistic regression-style output layer
One shared hidden layer with 2000 ReLU hidden units
Dropout probability of 0.1
Weight updates: SGD with Adam update
Grid search for hyperparameter tuning
Learning rate (best 1E-2)
Dropout (0.1, 0.3, 0.5)
Number of units in hidden layer (500, 1000, 2000, 4000)
Limited experimentation with number of hidden layers
Did not experiment extensively with learning rate schedule
"""

"""
Final Validation score: 
categorical_crossentropy, accuracy,         fmeasure,           precision,              recall
[0.13488368553714827, 0.38450000000000001, 0.53453228473663328, 0.38450000000000001, 0.95653295993804932]
Full data
categorical_crossentropy, accuracy, fmeasure, precision, recall
 [0.069954483653659971, 0.2880296532976121, 0.38409679690127507, 0.29799442753258215, 0.64157140408473257]

"""
