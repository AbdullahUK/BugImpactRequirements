"""Author:Abdullah Alsaeedi
Copyright (c) 2021 TU-CS Software Engineering Research Group (SERG),
Date: 22/03/2021
Name: Software Bug Severity using Machine Learning and Deep Learning
Version: 1.0
"""

import sys
import time
import re
import nltk
import string

def removeUnicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    return text


def replaceMultiExclamationMark(text):
    """ Replaces repetitions of exlamation marks """
    text = re.sub(r"(\!)\1+", '', text)
    return text

def replaceMultiQuestionMark(text):
    """ Replaces repetitions of question marks """
    text = re.sub(r"(\?)\1+", '', text)
    return text

def replaceMultiStopMark(text):
    """ Replaces repetitions of stop marks """
    text = re.sub(r"(\.)\1+", '', text)
    return text

# Processing texts
def preprocessTexts(text, stop_words):
    # Convert to lower case
    text = text.lower()

    # Convert www.* or https?://* to URL
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', text)

    # Convert @username to __HANDLE
    text = re.sub('@[^\s]+', '', text)

    # Replace #word with word
    text = re.sub(r'#([^\s]+)', r'\1', text)

    # trim
    text = text.strip('\'"')

    #print(string.punctuation)
    # Removing Puncations
    text = re.sub(r'[^\w\s]', '', text)

    # # Repeating words like happyyyyyyyy
    rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE)
    text = rpt_regex.sub(r"\1\1", text)

    #removing Numbers
    text =''.join(i for i in text if not i.isdigit())

    # Emoticons
    emoticons = \
        [
            (' positive ', [':-)', ': )',':)', '(:', '(-:', \
                              ':-D', ':D', 'X-D', 'XD', 'xD', \
                              '<3', ':\*', ';-)', ';)', ';-D', ';D', '(;', '(-;', ]), \
            (' negative ', [':-(', ': (',':(', '(:', '(-:', ':,(', \
                              ':\'(', ':"(', ':((', ]), \
            ]

    def replace_parenth(arr):
        return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]

    def regex_join(arr):
        return '(' + '|'.join(arr) + ')'

    emoticons_regex = [(repl, re.compile(regex_join(replace_parenth(regx)))) \
                       for (repl, regx) in emoticons]

    for (repl, regx) in emoticons_regex:
        text = re.sub(regx, ' ' + repl + ' ', text)

    # stopword Removal
    text = ' '.join(w for w in text.split() if w not in stop_words)

    # print(text)
    # text=replaceMultiQuestionMark(text)
    # text=replaceMultiStopMark(text)
    # text=replaceMultiExclamationMark(text)
    return text



# Stemming of Texts
def stem(text):
    stemmer = nltk.stem.PorterStemmer()
    text_stem = ''
    words = [word for word in text.split()]
    words = [stemmer.stem(w) for w in words]
    text_stem = ' '.join(words)
    return text_stem
