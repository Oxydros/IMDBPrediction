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
