#!/usr/bin/python

import os
import keras

from dataImporter import importData, sanitizeData

from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model, model_selection
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop

DATA_DIR = os.path.abspath("../data")

GLOBAL_DATA = os.path.join(DATA_DIR, "mix.csv")

def retrieveLabelAndFeatures(data, ids):
    #Logistic regression only take integer value as label
    #So we switch from 0.0 - 10.0 to 0 to 100
    #nb: LabelEncoder might be better
    labels = [int(float(data[i]["vote_average"]) * 10) for i in ids]
    features = [data[i] for i in ids]
    for f in features:
        f.pop("vote_average")
    print(features[0])
    #CF http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html
    v = DictVectorizer(sparse=False)
    features = v.fit_transform(features)
    print(v.get_feature_names())
    print(len(features[0]))
    return labels, features

def initData():
    print()
    print("Data dir: %s"%(DATA_DIR))
    print()
    print("Global file: %s"%(GLOBAL_DATA))
    result = importData(GLOBAL_DATA)
    result = sanitizeData(result)
    print()
    print("File imported. %d films"%(len(result.keys())))
    ids = list(result.keys())[:30000]

    #Retrieve labels and features
    labels, features = retrieveLabelAndFeatures(result, ids)

    numberOfIds = len(ids)
    trainingSize = 10000
    testingSize = numberOfIds - trainingSize
    print("Training size: %d    |   Testing size: %d"%(trainingSize, testingSize))

    featureTraining = features[:trainingSize]
    labelTraining = labels[:trainingSize]

    featureTesting = features[trainingSize + 1:]
    labelTesting = labels[trainingSize + 1:]
    return (featureTraining, labelTraining, featureTesting, labelTesting)

def LogisticRegressionModel(Xtrain, Ytrain, Xtest, Ytest):
    #Call logistic regression algorithm
    #CF http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
    model_LogReg = linear_model.LogisticRegression(max_iter=100, verbose=0)

    print("Training....")
    model_LogReg.fit(Xtrain, Ytrain)

    # accuracyTrain = model_selection.cross_val_score(model_LogReg, Xtrain, Ytrain)
    # accuracyTest = model_selection.cross_val_score(model_LogReg, Xtest, Ytest)
    # print("Cross val on train %s\nCross val on test %s"%(accuracyTrain, accuracyTest))

    yFound = model_LogReg.predict(Xtest)
    accuracyScore = accuracy_score(yFound, Ytest)
    print("Accuracy score: %s"%(accuracyScore))

def SequentialModel(Xtrain, Ytrain, Xtest, Ytest):
    model = Sequential()
    nbFeatures = len(Xtrain[0])
    epochs = 10
    batch_size = 256

    Ytrain = keras.utils.to_categorical(Ytrain)
    Ytest = keras.utils.to_categorical(Ytest)

    nbOutput = len(Ytrain[0])

    model.add(Dense(nbFeatures, activation='relu',
                    input_shape=(nbFeatures,)))
    model.add(Dropout(0.2))
    model.add(Dense(nbFeatures * 2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nbFeatures * 4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nbFeatures * 6, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nbFeatures * 4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nbFeatures * 2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nbOutput, activation='softmax'))

    print(model.summary())

    optimizer = RMSprop(lr=0.001)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(Xtrain, Ytrain, epochs=epochs,
              batch_size=batch_size,
              verbose=1,
              validation_data=(Xtest, Ytest))
    score = model.evaluate(Xtest, Ytest, batch_size=batch_size)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == "__main__":
    print("Movie recommendation engine - Predict score on IMDB")

    Xtrain, Ytrain, Xtest, Ytest = initData()
    # LogisticRegressionModel(Xtrain, Ytrain, Xtest, Ytest)
    SequentialModel(Xtrain, Ytrain, Xtest, Ytest)
