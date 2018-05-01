#!/usr/bin/python

import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from dataImporter import importData, sanitizeData

from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model, model_selection
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import validation_curve, learning_curve, ShuffleSplit
from sklearn.svm import SVC

from CNN_Models import SequentialCNNModel
from RNN_Models import SequentialRNNModel

DATA_DIR = os.path.abspath("../data")

GLOBAL_DATA = os.path.join(DATA_DIR, "mix.csv")

## From sklearn website
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def retrieveLabelAndFeatures(data, ids):
    #Logistic regression only take integer value as label
    #So we switch from 0.0 - 10.0 to 0 to 100
    #nb: LabelEncoder might be better
    labels = [float(data[i]["vote_average"]) for i in ids]
    features = [data[i] for i in ids]
    for f in features:
        f.pop("vote_average")
    #CF http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html
    v = DictVectorizer(sparse=False)
    features = v.fit_transform(features)
    # print(v.get_feature_names())
    print(features[0])
    print(len(features[0]))
    print(len(features))
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
    ids = list(result.keys())

    #Retrieve labels and features
    labels, features = retrieveLabelAndFeatures(result, ids)

    numberOfIds = len(ids)
    trainingSize = 40000
    testingSize = numberOfIds - trainingSize
    print("Training size: %d    |   Testing size: %d"%(trainingSize, testingSize))

    featureTraining = features[:trainingSize]
    labelTraining = labels[:trainingSize]

    featureTesting = features[trainingSize + 1:]
    labelTesting = labels[trainingSize + 1:]
    return (featureTraining, labelTraining, featureTesting, labelTesting)

def SVM(Xtrain, Ytrain, Xtest, Ytest):
    #Call logistic regression algorithm
    #CF http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
    model_SVC = SVC(verbose=0)

    title = "Learning curve SVC"
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    plot_learning_curve(model_SVC, title, Xtrain, Ytrain, ylim=(0.0, 1.0), cv=cv, n_jobs=-1)
    plt.savefig("LearningCurve_SVC.png")
    plt.clf()

def LogisticRegressionModel(Xtrain, Ytrain, Xtest, Ytest):
    #Call logistic regression algorithm
    #CF http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
    model_LogReg = linear_model.LogisticRegression(verbose=0, n_jobs=-1)

    title = "Learning curve Logistic Regression"
    cv = ShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
    plot_learning_curve(model_LogReg, title, Xtrain, Ytrain, ylim=(0.0, 1.0), cv=cv, n_jobs=-1)
    plt.savefig("LearningCurve_LogReg.png")
    plt.clf()

    print("Training LogisticRegression....")
    model_LogReg.fit(Xtrain, Ytrain)

    # accuracyTrain = model_selection.cross_val_score(model_LogReg, Xtrain, Ytrain)
    # accuracyTest = model_selection.cross_val_score(model_LogReg, Xtest, Ytest)
    # print("Cross val on train %s\nCross val on test %s"%(accuracyTrain, accuracyTest))
    print("Predict...")
    yFound = model_LogReg.predict(Xtest)
    # accuracyScore = accuracy_score(yFound, Ytest)
    # print("Accuracy score LogisticRegression: %s"%(accuracyScore))
    mserror = mean_squared_error(yFound, Ytest)
    print("Mean squared error score LogisticRegression: %s"%(mserror))

def RandomForestModel(Xtrain, Ytrain, Xtest, Ytest):
    #Call Random Forest algorithm
    #CF http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    # https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/
    model_RF = RandomForestRegressor(n_estimators=128, max_depth=None, max_features="auto", verbose=0, n_jobs=-1)

    title = "Learning curve Random Forest Regressor"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plot_learning_curve(model_RF, title, Xtrain, Ytrain, ylim=(0.0, 1.0), cv=cv, n_jobs=-1)
    plt.savefig("LearningCurve_RandFor.png")
    plt.clf()

    print("Training RandomForest....")
    model_RF.fit(Xtrain, Ytrain)

    print("Predict...")
    yFound = model_RF.predict(Xtest)

    # accuracyScore = accuracy_score(yFound, Ytest)
    # print("Accuracy score RandomForest: %s"%(accuracyScore))
    mserror = mean_squared_error(yFound, Ytest)
    print("Mean squared score RandomForest: %s"%(mserror))

if __name__ == "__main__":
    print("Movie recommendation engine - Predict score on IMDB")

    Xtrain, Ytrain, Xtest, Ytest = initData()
    LogisticRegressionModel(Xtrain, Ytrain, Xtest, Ytest)
    # RandomForestModel(Xtrain, Ytrain, Xtest, Ytest)
    # SVM(Xtrain, Ytrain, Xtest, Ytest)
    # SequentialCNNModel(Xtrain, Ytrain, Xtest, Ytest)
    # SequentialRNNModel(Xtrain, Ytrain, Xtest, Ytest)
