"""Authors:Abdullah Alsaeedi and Sultan Almaghdhui
Copyright (c) 2021 TU-CS Software Engineering Research Group (SERG),
Date: 22/03/2021
Name: Software Bug Severity using Machine Learning and Deep Learning
Version: 1.0
"""

# Import required libraries

import csv

def ReadBugSeverityDataset(DataSetName):
    X = []
    y = []
    with open('BugSeveritySEAA2017Dataset/'+DataSetName+'.csv', 'r', encoding='ISO-8859-1') as r:
        next(r)  # skip headers
        reader = csv.reader(r)
        for row in reader:
            X.append(row[0])
            if row[1] == "L0":
                y.append(0)
            elif row[1] == "L1":
                y.append(1)
            elif row[1] == "L2":
                y.append(2)
            elif row[1] == "L3":
                y.append(3)
            elif row[1] == "L4":
                y.append(4)
            elif row[1] == "L5":
                y.append(5)

    return X, y


def ReadEclipsedataset(DataSetName):
    X = []
    with open('DataSets/'+DataSetName+'.csv', 'r', encoding='ISO-8859-1') as r:
        next(r)  # skip headers
        reader = csv.reader(r)
        for row in reader:
            print(row[6])
            X.append(row[6])
    return X


