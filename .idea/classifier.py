import csv
import numpy as np
import sklearn.metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from os import walk
import string
import random
from nltk import word_tokenize
from nltk.corpus import stopwords, sentiwordnet as swn
from useful_functions_and_modules import create_vectorizer_and_classifier, classify_data
import scipy.stats
import statsmodels.stats.proportion

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

        if review_score < 10:
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
    lex_prec, lex_rec, lex_f = get_result(test.lexicon_label, test.label)
    clas_prec, clas_rec, cla_f = get_result(test.classifier_label, test.label)

    print("For the lexicon classifier we have precision " + str(lex_prec) + ", recall " + str(lex_rec) + ", fscore " + str(lex_f))
    print("For the other classifier we have precision " + str(clas_prec) + ", recall " + str(clas_rec) + ", fscore " + str(cla_f))

    #For the lexicon classifier we have precision [0.7 0.51081081], recall [0.1039604  0.95454545], fscore [0.18103448 0.66549296]
    #For the other classifier we have precision [0.82828283 0.81188119], recall [0.81188119 0.82828283], fscore [0.82 0.82]

    test['lex_correct'] = [1 if x['label'] == x['lexicon_label'] else 0 for _, x in test.iterrows()]
    test['clas_correct'] = [1 if x['label'] == x['classifier_label'] else 0 for _, x in test.iterrows()]

    # Test with H0: there is no difference for lexicon based classifier and the other. gives pvalue 4.7582126794873315e-130
    mean_lex = np.average(test['lex_correct'])
    mean_reg = np.average(test['clas_correct'])
    pvalue = scipy.stats.binom_test(mean_reg, test.shape[0], mean_lex)
    print("p-values is " + str(pvalue))

    # This test is testing the wrong thing I think, it gives p-value 2.949704220103885e-10
    # test['agreement'] = [1 if x['lexicon_label'] == x['classifier_label'] else 0 for _, x in test.iterrows()]
    count = 400 - abs(sum(test['clas_correct']) - sum(test['lex_correct']))
    pvalue2 = statsmodels.stats.proportion.binom_test(count, 400, prop=0.5, alternative='two-sided')
    print("Or is the pvalue: " + str(pvalue2))


def get_result(y_pred, y_true):
    # Gets precision, recal and f score of a classifier
    prec_rec_fscore = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred)
    precision = prec_rec_fscore[0]
    recall = prec_rec_fscore[1]
    fbeta_score = prec_rec_fscore[2]

    return [precision, recall, fbeta_score]

def get_scores(token):
    # Gets the score of a token from the lexicon
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
    # Tokenize a text
    tokens = word_tokenize(text)

    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]

    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]

    return tokens

## This was the code I spend 6 hours on because I didn't know there was a function for getting the score.
## from the lexicon.

# def get_scores_of_words():
#     with open("SentiLexiconFromSentiWordNet.txt", 'r') as lexicon:
#         csv_read = csv.reader(lexicon, delimiter='\t')
#         pos_score = []
#         neg_score = []
#         words = {}
#         line_count = 0
#         for row in csv_read:
#             if line_count > 0:
#                 word = row[4]
#                 pos_score = row[2]
#                 neg_score = row[3]
#                 if ' ' in word:
#                     wds = word.split(" ")
#                     for w in wds:
#                         split_word = w.split("#", 1)[0]
#                         if split_word not in words:
#                             words[split_word] = [pos_score, neg_score]
#                 else:
#                     split_word = word.split("#", 1)[0]
#                     if split_word not in words:
#                         words[split_word] = [pos_score, neg_score]
#             line_count += 1
#         return words

classifier()
