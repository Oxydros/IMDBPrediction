#!/usr/bin/python

import os

from dataImporter import importData

DATA_DIR = os.path.abspath("../data")

GLOBAL_DATA = os.path.join(DATA_DIR, "mix.csv")

if __name__ == "__main__":
    print("Movie recommendation engine - Predict score on IMDB")
    print()
    print("Data dir: %s"%(DATA_DIR))
    print()
    print("Global file: %s"%(GLOBAL_DATA))
    result = importData(GLOBAL_DATA)
    print()
    print("File imported. %d films"%(len(result.keys())))
