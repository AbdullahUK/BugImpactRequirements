"""Author:Abdullah Alsaeedi
Copyright (c) 2021 TU-CS Software Engineering Research Group (SERG),
Date: 22/03/2021
Name: Software Bug Severity using Machine Learning and Deep Learning
Version: 1.0
"""

# Import required libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import KFold, GridSearchCV

from LoadingData import *
from Preprocessing import *
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from nltk.corpus import stopwords
from Learners import *
from sklearn.decomposition import NMF, LatentDirichletAllocation
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def PreprocessBugsDescription(X_train, X_test):
    stop_words = set(stopwords.words('english'))
    X_train = [stem(preprocessTexts(text, stop_words)) for text in X_train]
    X_test = [stem(preprocessTexts(text, stop_words)) for text in X_test]
    return X_train, X_test

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for Dataset in ['bugs-2021-03-15 - eclipse modeling']:
        print(Dataset)
        X= ReadEclipsedataset(Dataset)
        stop_words = set(stopwords.words('english'))
        X = [preprocessTexts(text, stop_words) for text in X]
        no_features = 1000

        # NMF is able to use tf-idf
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(X)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()

        # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
        tf = tf_vectorizer.fit_transform(X)
        tf_feature_names = tf_vectorizer.get_feature_names()

        no_topics = 6

        # Run NMF
        nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

        # Run LDA
        lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,
                                        random_state=0).fit(tf)

        no_top_words = 10
        display_topics(nmf, tfidf_feature_names, no_top_words)
        display_topics(lda, tf_feature_names, no_top_words)

        # X = pd.np.asarray(X)
        # y = pd.np.asarray(y)
        #
        # total_Accuracy_score_array = dict()
        # total_Precision_score_array = dict()
        # total_Recall_score_array = dict()
        # total_Fscore_score_array = dict()
        # kf = KFold(n_splits=5, shuffle=True)  # 5 fold similar to (Arabic Sentiment Analysis:Lexicon-based and Corpus-based) paper
        # for train_index, test_index in kf.split(X):
        #     X_train, X_test = X[train_index], X[test_index]
        #     y_train, y_test = y[train_index], y[test_index]
        #     X_train, X_test = PreprocessBugsDescription(X_train, X_test)
        #
        #     vec = TfidfVectorizer(ngram_range=(1, 3))
        #     X_train_tf_idf = vec.fit_transform(X_train, y_train)
        #     print(X_train_tf_idf)
        #     X_test_tf_idf = vec.transform(X_test)
        #     n_features = X_train_tf_idf.shape[1]
        #     for cls in ['RF', 'DS', 'SVM', 'LR']:
        #         # Accuracy_score_array = []
        #         # Precision_score_array = []
        #         # Recall_score_array = []
        #         # Fscore_score_array = []
        #         # AUC_score_array = []
        #         print("-----------------------------------------------------------")
        #         print('CurrentClassifier= ' + cls)
        #         if cls == 'RF':
        #             accuracy, precision, recall, fscore = RandomForestLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test)
        #         elif cls == 'DS':
        #             accuracy, precision, recall, fscore = DecisionTreeLearner(X_train_tf_idf, X_test_tf_idf,y_train,y_test)
        #         elif cls == 'SVM':
        #             accuracy, precision, recall, fscore = SVMLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test)
        #         elif cls == 'MNB':
        #             accuracy, precision, recall, fscore = MultinomialNBLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test)
        #         elif cls == 'LR':
        #             accuracy, precision, recall, fscore = LogisticRegressionLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test)
        #         elif cls == 'BNB':
        #             accuracy, precision, recall, fscore = BernoulliNBLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test)
        #         elif cls == 'AdaBoost':
        #             accuracy, precision, recall, fscore = AdaBoostLearner(X_train_tf_idf, X_test_tf_idf, y_train,y_test,n_features)
        #
        #         print('accuracy= ', accuracy)
        #         total_Accuracy_score_array.setdefault(cls, []).append(accuracy)
        #         total_Precision_score_array.setdefault(cls, []).append(precision)
        #         total_Recall_score_array.setdefault(cls, []).append(recall)
        #         total_Fscore_score_array.setdefault(cls, []).append(fscore)
        #
        #     for cls in ['RF', 'DS', 'SVM', 'LR']:
        #         print("AccuracyScore for " + cls + " classifier=  ",
        #               pd.np.mean(total_Accuracy_score_array.get(cls), axis=0))
        #         print("PrecisionScore for " + cls + " classifier=  ",
        #               pd.np.mean(total_Precision_score_array.get(cls), axis=0))
        #         print("RecallScore for " + cls + " classifier=  ",
        #               pd.np.mean(total_Recall_score_array.get(cls), axis=0))
        #         print("FscoreScore for " + cls + " classifier=  ",
        #               pd.np.mean(total_Fscore_score_array.get(cls), axis=0))

