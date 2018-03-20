#!/usr/bin/python

import os
import csv

DATA_DIR = os.path.abspath("../data")
CREDITS = os.path.join(DATA_DIR, "credits.csv")
MOVIES_METADATA = os.path.join(DATA_DIR, "movies_metadata.csv")
KEYWORDS = os.path.join(DATA_DIR, "keywords.csv")

GLOBAL_DATA = os.path.join(DATA_DIR, "mix.csv")

def importCSV(filePath):
    result = []
    with open(filePath) as fd:
        data = csv.DictReader(fd)
        result = [i for i in data]
    return (result)

def build_dict(seq, key):
    return (dict((d[key], dict(d, index=index)) for (index, d) in enumerate(seq)))

def importData():
    credits = importCSV(CREDITS)
    metadata = importCSV(MOVIES_METADATA)
    keywords = importCSV(KEYWORDS)
    creditsById = build_dict(credits, "id")
    metadataById = build_dict(metadata, "id")
    keywordsById = build_dict(keywords, "id")
    result = []
    for idValue in creditsById.keys():
        m = metadataById[idValue]
        k = keywordsById[idValue]
        c = creditsById[idValue]
        newDic = {**m, **k, **c}
        result.append(newDic)
    return (result)

def exportData(data):
    with open(GLOBAL_DATA, 'w') as fd:
        writer = csv.DictWriter(fd, fieldnames=data[0].keys())
        writer.writeheader()
        for d in data:
            writer.writerow(d)

if __name__ == "__main__":
    print("Movie recommendation engine - Predict score on IMDB")
    print()
    print("Data dir: %s"%(DATA_DIR))
    print("Credits: %s"%(CREDITS))
    print("Movies metadata %s"%(MOVIES_METADATA))
    print("Keywords: %s"%(KEYWORDS))
    print()
    print("Mixing everything by id. Output: %s"%(GLOBAL_DATA))
    result = importData()
    exportData(result)
    print()
    print("Export completed !")
    # a = importCSV(GLOBAL_DATA)
    # a = build_dict(a, "id")
    # print(a["862"])
