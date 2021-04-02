"""Authors:Abdullah Alsaeedi and Sultan Almaghdhui
Copyright (c) 2021 TU-CS Software Engineering Research Group (SERG),
Date: 22/03/2021
Name: Software Bug Severity using Machine Learning and Deep Learning
Version: 1.0
"""

# Import required libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from LoadingData import *
from Preprocessing import *
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from nltk.corpus import stopwords
from Learners import *
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def PreprocessBugsDescription(X_train, X_test):
    stop_words = set(stopwords.words('english'))
    X_train = [stem(preprocessTexts(text, stop_words)) for text in X_train]
    X_test = [stem(preprocessTexts(text, stop_words)) for text in X_test]
    return X_train, X_test


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for Dataset in ['mongo-db']:
        print(Dataset)
        X, y = ReadBugSeverityDataset(Dataset)
        X = pd.np.asarray(X)
        y = pd.np.asarray(y)

        total_Accuracy_score_array = dict()
        total_Precision_score_array = dict()
        total_Recall_score_array = dict()
        total_Fscore_score_array = dict()
        kf = KFold(n_splits=5, shuffle=True)  # 5 fold similar to (Arabic Sentiment Analysis:Lexicon-based and Corpus-based) paper
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train, X_test = PreprocessBugsDescription(X_train, X_test)

            vec = TfidfVectorizer(ngram_range=(1, 3))
            X_train_tf_idf = vec.fit_transform(X_train, y_train)
            print(X_train_tf_idf)
            X_test_tf_idf = vec.transform(X_test)
            n_features = X_train_tf_idf.shape[1]
            for cls in ['RF', 'DS', 'SVM', 'LR']:
                # Accuracy_score_array = []
                # Precision_score_array = []
                # Recall_score_array = []
                # Fscore_score_array = []
                # AUC_score_array = []
                print("-----------------------------------------------------------")
                print('CurrentClassifier= ' + cls)
                if cls == 'RF':
                    accuracy, precision, recall, fscore = RandomForestLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test)
                elif cls == 'DS':
                    accuracy, precision, recall, fscore = DecisionTreeLearner(X_train_tf_idf, X_test_tf_idf,y_train,y_test)
                elif cls == 'SVM':
                    accuracy, precision, recall, fscore = SVMLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test)
                elif cls == 'MNB':
                    accuracy, precision, recall, fscore = MultinomialNBLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test)
                elif cls == 'LR':
                    accuracy, precision, recall, fscore = LogisticRegressionLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test)
                elif cls == 'BNB':
                    accuracy, precision, recall, fscore = BernoulliNBLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test)
                elif cls == 'AdaBoost':
                    accuracy, precision, recall, fscore = AdaBoostLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test,n_features)

                print('accuracy= ', accuracy)
                total_Accuracy_score_array.setdefault(cls, []).append(accuracy)
                total_Precision_score_array.setdefault(cls, []).append(precision)
                total_Recall_score_array.setdefault(cls, []).append(recall)
                total_Fscore_score_array.setdefault(cls, []).append(fscore)

            for cls in ['RF', 'DS', 'SVM', 'LR']:
                print("AccuracyScore for " + cls + " classifier=  ",
                      pd.np.mean(total_Accuracy_score_array.get(cls), axis=0))
                print("PrecisionScore for " + cls + " classifier=  ",
                      pd.np.mean(total_Precision_score_array.get(cls), axis=0))
                print("RecallScore for " + cls + " classifier=  ",
                      pd.np.mean(total_Recall_score_array.get(cls), axis=0))
                print("FscoreScore for " + cls + " classifier=  ",
                      pd.np.mean(total_Fscore_score_array.get(cls), axis=0))

