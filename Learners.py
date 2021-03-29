"""Author:Abdullah Alsaeedi
Copyright (c) 2021 TU-CS Software Engineering Research Group (SERG),
Date: 22/03/2021
Name: Software Bug Severity using Machine Learning and Deep Learning
Version: 1.0
"""

# Import required libraries
import sklearn.metrics

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from tensorflow.python.keras import callbacks, Input, Model, Sequential
from tensorflow.python.keras.layers import Embedding, SpatialDropout1D, LSTM, Conv1D, GlobalMaxPooling1D, Concatenate, \
    Dense, MaxPooling1D, Flatten, Bidirectional
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer


def RandomForestLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test):
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(X_train_tf_idf, y_train)
    y_pred = classifier.predict(X_test_tf_idf)
    print(sklearn.metrics.confusion_matrix(y_test, y_pred))
    return sklearn.metrics.accuracy_score(y_test, y_pred),sklearn.metrics.precision_score(y_test, y_pred, average='weighted'),\
           sklearn.metrics.recall_score(y_test, y_pred, average='weighted'), sklearn.metrics.f1_score(y_test, y_pred, average='weighted')

def DecisionTreeLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test):
    classifier = DecisionTreeClassifier(criterion='entropy')
    classifier.fit(X_train_tf_idf, y_train)
    y_pred = classifier.predict(X_test_tf_idf)
    return sklearn.metrics.accuracy_score(y_test, y_pred), sklearn.metrics.precision_score(y_test, y_pred, average='weighted'),\
           sklearn.metrics.recall_score(y_test, y_pred, average='weighted'), sklearn.metrics.f1_score(y_test, y_pred, average='weighted')

def SVMLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test):
    classifier = LinearSVC(C=0.1)
    classifier.fit(X_train_tf_idf, y_train)
    y_pred = classifier.predict(X_test_tf_idf)
    return sklearn.metrics.accuracy_score(y_test, y_pred), sklearn.metrics.precision_score(y_test, y_pred,average='weighted'),\
           sklearn.metrics.recall_score(y_test, y_pred, average='weighted'), sklearn.metrics.f1_score(y_test, y_pred, average='weighted')

def  MultinomialNBLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test):
    classifier = MultinomialNB()
    classifier.fit(X_train_tf_idf, y_train)
    y_pred = classifier.predict(X_test_tf_idf)
    return sklearn.metrics.accuracy_score(y_test, y_pred), sklearn.metrics.precision_score(y_test, y_pred,average='weighted'), \
           sklearn.metrics.recall_score(y_test, y_pred, average='weighted'), sklearn.metrics.f1_score(y_test,y_pred,average='weighted')

def LogisticRegressionLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test):
    classifier = LogisticRegression(solver='lbfgs')
    classifier.fit(X_train_tf_idf, y_train)
    y_pred = classifier.predict(X_test_tf_idf)
    return sklearn.metrics.accuracy_score(y_test, y_pred), sklearn.metrics.precision_score(y_test, y_pred, average='weighted'), \
           sklearn.metrics.recall_score(y_test, y_pred, average='weighted'), sklearn.metrics.f1_score(y_test, y_pred,average='weighted')

def BernoulliNBLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test):
    classifier = BernoulliNB()
    classifier.fit(X_train_tf_idf, y_train)
    y_pred = classifier.predict(X_test_tf_idf)
    return sklearn.metrics.accuracy_score(y_test, y_pred), sklearn.metrics.precision_score(y_test, y_pred, average='weighted'), \
           sklearn.metrics.recall_score(y_test, y_pred, average='weighted'), sklearn.metrics.f1_score(y_test, y_pred,average='weighted')

def AdaBoostLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test,n_features):
    classifier = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy'), algorithm="SAMME", n_estimators=100)
    classifier.fit(X_train_tf_idf, y_train)
    y_pred = classifier.predict(X_test_tf_idf)
    return sklearn.metrics.accuracy_score(y_test, y_pred), sklearn.metrics.precision_score(y_test, y_pred,average='weighted'), \
           sklearn.metrics.recall_score(y_test, y_pred, average='weighted'), sklearn.metrics.f1_score(y_test, y_pred,average='weighted')

