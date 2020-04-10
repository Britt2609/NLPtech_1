import csv
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import os
from os import walk
import random
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.corpus import sentiwordnet as swn
from useful_functions_and_modules import create_vectorizer_and_classifier, classify_data


def classifier():
    dataframe = get_df("review_polarity_data")
    train, test = train_test_split(dataframe, test_size=0.2)
    all_review_scores = []
    review_labels = []
    num_neg = 0

    for _, row in test.iterrows():
        tokens = tokenize(row.text)
        review_score = 0

        for token in tokens:
            token_score = get_scores(token)
            review_score += token_score[2]

        if review_score < 2:
            review_labels.append("neg")
            num_neg += 1
        else:
            review_labels.append("pos")
        all_review_scores.append(review_score)

    # Using the given methods
    vectorizer, classifier = create_vectorizer_and_classifier(train.text, train.label)
    result_classifier = classify_data(test.text, vectorizer, classifier)

    test['lexicon_label'] = review_labels
    test['classifier_label'] = result_classifier
    print(get_result(test.lexicon_label, test.label))
    print(get_result(test.classifier_label, test.label))

def get_result(y_pred, y_true):
    # This function returns:
    # precision float (if average is not None) or array of float, shape = [n_unique_labels]
    # recall float (if average is not None) or array of float, , shape = [n_unique_labels]
    # fbeta_score float (if average is not None) or array of float, shape = [n_unique_labels]
    # support None (if average is not None) or array of int, shape = [n_unique_labels]
    return sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, labels=['neg', 'pos'])

def get_scores(token):
    pos_score = 0
    neg_score = 0
    for syn in swn.senti_synsets(token):
        pos_score += syn.pos_score()
        neg_score += syn.neg_score()
    return [pos_score, neg_score, pos_score - neg_score]

def get_df(directory):
    # read data as Pandas DataFrame
    df = []
    labels = [x[1] for x in os.walk(directory)][0]
    data = [x[2] for x in os.walk(directory) if x[2] != []]
    counter = 0

    for i in range(len(labels)):
        for x in data[i]:
            full_dir = directory + '/' + labels[i] + '/' + x
            with open(full_dir, 'r') as file:
                dat = file.read().replace('\n', '')

            d = {'text': dat, 'label': labels[i]}
            df.append(d)

    # Final DataFrame to use
    return pd.DataFrame(df)

def tokenize(text):

    tokens = word_tokenize(text)

    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]

    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]

    return tokens

## This was the code I spend 6 hours on because I didn't know there was a function for this.
def get_scores_of_words():
    with open("SentiLexiconFromSentiWordNet.txt", 'r') as lexicon:
        csv_read = csv.reader(lexicon, delimiter='\t')
        pos_score = []
        neg_score = []
        words = {}
        line_count = 0
        for row in csv_read:
            if line_count > 0:
                word = row[4]
                pos_score = row[2]
                neg_score = row[3]
                if ' ' in word:
                    wds = word.split(" ")
                    for w in wds:
                        split_word = w.split("#", 1)[0]
                        if split_word not in words:
                            words[split_word] = [pos_score, neg_score]
                else:
                    split_word = word.split("#", 1)[0]
                    if split_word not in words:
                        words[split_word] = [pos_score, neg_score]
            line_count += 1
        return words

classifier()
