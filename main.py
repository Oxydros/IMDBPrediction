#!/usr/bin/python

import os

import json
import re
from dataImporter import importData

from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model, model_selection

DATA_DIR = os.path.abspath("../data")

GLOBAL_DATA = os.path.join(DATA_DIR, "mix.csv")

LABEL_NAME = "vote_average"

FEATURES = ["adult", "budget", "genres",
            #"original_language",
            # "production_companies", "production_countries",
            #"spoken_languages", #"keywords",
            "revenue", "runtime"]

GENRES_VALUES = set()

def sanitizeData(data):
    wantedFields = FEATURES
    toDelete = []
    for idValue in data.keys():
        k = data[idValue].keys()

        #Check wanted fields
        for wanted in wantedFields:
            if not len(data[idValue][wanted]):
                toDelete.append(idValue)
                break

        #Remove unwanted features:
        keys = list(data[idValue].keys())
        for feature in keys:
            if not feature in FEATURES and not feature in LABEL_NAME:
                data[idValue].pop(feature)

        #Sanitize dicts of multiple string value, keeping only the "name" field
        for stringDic in [
                          "genres"
                         # "spoken_languages"
                          #"keywords"
                          ]: #"production_companies", "production_countries"
            findName = re.compile("'name':( '.*?')")
            findName = findName.findall(data[idValue][stringDic])
            for name in findName:
                name = name.strip().replace("'", '')
                data[idValue][name] = 1
            data[idValue].pop(stringDic)

        #Sanitize strings, keeping only the value
        # for stringDic in ["original_language"]:
        #     value = data[idValue][stringDic]
        #     data[idValue][value] = 1
        #     data[idValue].pop(stringDic)

        #Hardcoded adult value
        data[idValue]["adult"] = 1 if data[idValue]["adult"] == "True" else 0

        for key in data[idValue]:
            try:
                data[idValue][key] = int(float(data[idValue][key]))
            except:
                toDelete.append(idValue)

    for i in toDelete:
        try:
            data.pop(i)
        except:
            pass
    return (data)

def retrieveLabelAndFeatures(data, ids):
    #Logistic regression only take integer value as label
    #So we switch from 0.0 - 10.0 to 0 to 100
    #nb: LabelEncoder might be better
    labels = [int(float(result[i]["vote_average"]) * 10) for i in ids]
    features = [result[i] for i in ids]
    for f in features:
        f.pop(LABEL_NAME)
    print(features[0])
    #CF http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html
    v = DictVectorizer(sparse=False)
    features = v.fit_transform(features)
    print(v.get_feature_names())
    print(len(features[0]))
    return labels, features

if __name__ == "__main__":
    print("Movie recommendation engine - Predict score on IMDB")
    print()
    print("Data dir: %s"%(DATA_DIR))
    print()
    print("Global file: %s"%(GLOBAL_DATA))
    result = importData(GLOBAL_DATA)
    result = sanitizeData(result)
    print()
    print("File imported. %d films"%(len(result.keys())))
    ids = list(result.keys())[:20000]

    #Retrieve labels and features
    labels, features = retrieveLabelAndFeatures(result, ids)

    numberOfIds = len(ids)
    trainingSize = 5000
    testingSize = numberOfIds - trainingSize
    print("Training size: %d    |   Testing size: %d"%(trainingSize, testingSize))

    featureTraining = features[:trainingSize]
    labelTraining = labels[:trainingSize]

    featureTesting = features[trainingSize + 1:]
    labelTesting = labels[trainingSize + 1:]

    #Call logistic regression algorithm
    #CF http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
    logRegreModel = linear_model.LogisticRegression(C=1e5)

    print("Training....")
    logRegreModel.fit(featureTraining, labelTraining)

    accuracy = model_selection.cross_val_score(logRegreModel, featureTesting, labelTesting)

    print(accuracy)
    # print("Testing....")
    # ret = logRegreModel.predict(featureTesting)
