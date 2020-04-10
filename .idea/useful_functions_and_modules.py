#scikit learn provides several machine learning modules. They follow the same general procedure.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
#easy to use tokenizer
from nltk import word_tokenize

# Other useful modules:
# sys (lightweight module for reading in arguments)
# random (module that provides functions for random selections and choices)
#
# it may be useful to represent the data in csv. In that case, it is recommended to use a specialized module for dealing with data in csv
# csv (lightweight, easy to use)
# pandas (offers more possibilities and is used in plenty tutorials; but can slow down your program a lot if not used in the right way)





def create_vectorizer_and_classifier(reviews, labels):
    '''
    This function creates a vectorizer and logistic regression classifer with bag-of-word features
    
    :param reviews: the review texts (scenario sentiment classification on reviews)
    :param labels: the gold labels for the texts
    :type reviews: list of strings (each element is the text of a review)
    :type labels: list of strings (each element provides the gold label for the corresponding review)
    
    :returns: the vectorizer that provides the mapping from tokens to a vector and a trained classifier
    '''
    
    vectorizer = CountVectorizer(min_df=1,tokenizer=word_tokenize)
    training_vector = vectorizer.fit_transform(reviews)
    myclassifier = LogisticRegression(max_iter=1000)
    myclassifier.fit(training_vector, labels)
    
    return vectorizer, myclassifier
    


def classify_data(evaluation_reviews,vectorizer, sentiment_classifier):
    '''
    This function takes a list of strings and labels them with a classifier
    
    :param evaluation_reviews: the reviews to be used for evaluation
    :param vectorizer: the vectorizer to turn text in bag-of-word vector representations
    :param sentiment_classifier: a trained classifier
    :type evaluation_reviews: list of strings
    :type vectorizer: CountVectorizer
    :type classifier: LogisticRegression() (can be another type of classifier as well)
    
    :returns: a list of predicted labels
    '''

    evaluation_vector = vectorizer.transform(evaluation_reviews)
    predictions = sentiment_classifier.predict(evaluation_vector)

    return predictions
