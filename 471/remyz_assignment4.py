# Skeleton for Assignment 4.
# Ling471 Spring 2021.

import pandas as pd
import string
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# These are your own functions you wrote for Assignment 3:
from evaluation import computePrecisionRecall, computeAccuracy


# Constants
ROUND = 4
GOOD_REVIEW = 1
BAD_REVIEW = 0
ALPHA = 1


# This function will be reporting errors due to variables which were not assigned any value.
def main(argv):
    data = pd.read_csv(argv[1])
    # print(data.head()) 

    test_data = data[:25000]  # Assuming the first 25,000 rows are test data.

    train_data = data[25000:50000]

    X_train = train_data["review"]
    y_train = train_data["label"]
    X_test = test_data["review"]
    y_test = test_data["label"]

    # the astype() method converts the datatype of a column in a dataframe, in this case converting int64 to int32
    # this will format the data so it can be used when we call other packages with this dataframe
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

   
    # discard non content words
    tf_idf_vect = TfidfVectorizer(ngram_range=(1, 2))
    tf_idf_train = tf_idf_vect.fit_transform(X_train.values)
    tf_idf_test = tf_idf_vect.transform(X_test.values)

    # laplace smoothing
    clf = MultinomialNB(alpha=ALPHA)

    #Naive Bayes classifier
    clf.fit(tf_idf_train, y_train)
     
    y_pred_train = clf.predict(tf_idf_train)
    y_pred_test = clf.predict(tf_idf_test)

    #calculate accuracy, precision, and recall
    accuracy_test = computeAccuracy(list(y_pred_test), list(y_test))
    accuracy_train = computeAccuracy(list(y_pred_train), list(y_train))
    precision_pos_test, recall_pos_test = computePrecisionRecall(list(y_pred_test), list(y_test), "POSITIVE")
    precision_neg_test, recall_neg_test = computePrecisionRecall(list(y_pred_test), list(y_test), "NEGATIVE")
    precision_pos_train, recall_pos_train = computePrecisionRecall(list(y_pred_train), list(y_train), "POSITIVE")
    precision_neg_train, recall_neg_train = computePrecisionRecall(list(y_pred_train), list(y_train), "NEGATIVE")

    print("Train accuracy:           \t{}".format(round(accuracy_train, ROUND)))
    print("Train precision positive: \t{}".format(
        round(precision_pos_train, ROUND)))
    print("Train recall positive:    \t{}".format(
        round(recall_pos_train, ROUND)))
    print("Train precision negative: \t{}".format(
        round(precision_neg_train, ROUND)))
    print("Train recall negative:    \t{}".format(
        round(recall_neg_train, ROUND)))
    print("Test accuracy:            \t{}".format(round(accuracy_test, ROUND)))
    print("Test precision positive:  \t{}".format(
        round(precision_pos_test, ROUND)))
    print("Test recall positive:     \t{}".format(
        round(recall_pos_test, ROUND)))
    print("Test precision negative:  \t{}".format(
        round(precision_neg_test, ROUND)))
    print("Test recall negative:     \t{}".format(
        round(recall_neg_test, ROUND)))


if __name__ == "__main__":
    main(sys.argv)
