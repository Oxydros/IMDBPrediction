import json
import re
import csv

def importCSV(filePath):
    result = []
    with open(filePath) as fd:
        data = csv.DictReader(fd)
        result = [i for i in data]
    return (result)

def build_dict(seq, key):
    return (dict((d[key], dict(d, index=index)) for (index, d) in enumerate(seq)))

def importData(path):
    data = importCSV(path)
    data = build_dict(data, "id")
    return (data)


LABEL_NAME = "vote_average"

FEATURES = ["adult", "budget",
            "genres",
            "popularity",
            #"original_language",
            #"production_companies", "production_countries",
            #"spoken_languages", #"keywords",
            "revenue", "runtime"]

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
                          "genres",
                          #"spoken_languages",
                          #"production_companies", "production_countries"
                          #"keywords"
                          ]:
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
