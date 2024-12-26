import pandas as pd
import string
import os
import sys
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import matplotlib.pyplot as plt

from evaluation import computeAccuracy, computePrecisionRecall
# Constants
ROUND = 4
GOOD_REVIEW = 1
BAD_REVIEW = 0
ALPHA = 1


def my_naive_bayes(column_name, data):
    
    test_data = data[:25000]  
    train_data = data[25000:50000]


 
    X_train = train_data[column_name]
    y_train = train_data["label"]
    X_test = test_data[column_name]
    y_test = test_data["label"]

    #convert datatype
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
 
    #vectorize data from reviews    
    tf_idf_vect = TfidfVectorizer(ngram_range=(1, 2))
    tf_idf_train = tf_idf_vect.fit_transform(X_train.values)
    tf_idf_test = tf_idf_vect.transform(X_test.values)
    #Smooth
    clf = MultinomialNB(alpha=ALPHA)
    #fit to Naive Bayes
    clf.fit(tf_idf_train, y_train)

    #make predictions using our trained vectors 
    y_pred_train = clf.predict(tf_idf_train)
    y_pred_test = clf.predict(tf_idf_test)

    # Compute accuracy, precision, and recall, for both train and test data.
    accuracy_test = computeAccuracy(list(y_pred_test), list(y_test))
    accuracy_train = computeAccuracy(list(y_pred_train), list(y_train))
    precision_pos_test, recall_pos_test = computePrecisionRecall(list(y_pred_test), list(y_test), "POSITIVE")
    precision_neg_test, recall_neg_test = computePrecisionRecall(list(y_pred_test), list(y_test), "NEGATIVE")
    precision_pos_train, recall_pos_train = computePrecisionRecall(list(y_pred_train), list(y_train), "POSITIVE")
    precision_neg_train, recall_neg_train = computePrecisionRecall(list(y_pred_train), list(y_train), "NEGATIVE")
    '''
    # Report the metrics 
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
    '''
    return({'TRAIN': {'accuracy': accuracy_train, 'POS': {'precision': precision_pos_train, 'recall': recall_pos_train}, 'NEG': {'precision': precision_neg_train, 'recall': recall_neg_train}}, 'TEST': {'accuracy': accuracy_test, 'POS': {'precision': precision_pos_test, 'recall': recall_pos_test}, 'NEG': {'precision': precision_neg_test, 'recall': recall_neg_test}}})

def main(argv):
    data = pd.read_csv("my_imdb_expanded.csv")

    
    #print('original')
    nb_original = my_naive_bayes('review', data)
    #print('cleaned')
    nb_cleaned = my_naive_bayes('cleaned_review', data)
    #print('lower')
    nb_lowercase = my_naive_bayes('lowercased', data)
    #print('stopword')
    nb_no_stop = my_naive_bayes('no stopwords', data)
    #print('lemmatized')
    nb_lemmatized = my_naive_bayes('lemmatized', data)

    train_accuracies = []
    test_accuracies = []
    train_pos_precision = []
    train_pos_recall = []
    train_neg_precision = []
    train_neg_recall = []
    test_pos_precision = []
    test_pos_recall = []
    test_neg_precision = []
    test_neg_recall = []

    for model in [nb_original, nb_cleaned, nb_lowercase, nb_no_stop, nb_lemmatized]:
        train_accuracies.append(model['TRAIN']['accuracy'])
        test_accuracies.append(model['TEST']['accuracy'])
        train_pos_precision.append(model['TRAIN']['POS']['precision'])
        train_pos_recall.append(model['TRAIN']['POS']['recall'])
        train_neg_precision.append(model['TRAIN']['NEG']['precision'])
        train_neg_recall.append(model['TRAIN']['NEG']['recall'])
        test_pos_precision.append(model['TEST']['POS']['precision'])
        test_pos_recall.append(model['TEST']['POS']['recall'])
        test_neg_precision.append(model['TEST']['NEG']['precision'])
        test_neg_recall.append(model['TEST']['NEG']['recall'])

  # create plots
    results = pd.DataFrame({
        "train accuracies": train_accuracies,
        "test_accuracies" : test_accuracies,
        "train pos precision" : train_pos_precision,
        "train pos recall" : train_pos_recall,
        "train neg precision" : train_neg_precision,
        "train neg recall" : train_neg_recall,
        "test pos precision" : test_pos_precision,
        "test pos recall" : test_pos_recall,
        "test neg precision" : test_neg_precision,
        "test neg recall" : test_neg_recall, 
        'index' : ["Original", "Cleaned", "Lowercase", "No_stop", "Lemmatized"]})

    #print(results)

    results.plot(x = 'index', y = ["train accuracies", "test_accuracies"], kind = "bar")
    plt.title("Comparing Accuracy of Different Models")
    plt.xlabel("Models")
    plt.ylabel("Accuracy Score")
    plt.ylim(0.8, 1)
    plt.savefig('accuracies.png')
        
    results.plot(x = 'index', y = ["train pos precision", "train pos recall"], kind = "bar")
    plt.title("Precision and Recall of Different Models \n (Evaluated on Train; Respect to Positive")
    plt.xlabel("Models")
    plt.ylabel("Precicions and Recall Score")
    plt.ylim(0.8, 1)
    plt.savefig('train_pos.png')

    results.plot(x = 'index', y = ["train neg precision", "train neg recall"], kind = "bar")
    plt.title("Precision and Recall of Different Models \n (Evaluated on Train; Respect to Negative")
    plt.xlabel("Models")
    plt.ylabel("Precicions and Recall Score")
    plt.ylim(0.8, 1)
    plt.savefig('train_neg.png')

    results.plot(x = 'index', y = ["test pos precision", "test pos recall"], kind = "bar")
    plt.title("Precision and Recall of Different Models \n (Evaluated on Test; Respect to Positive")
    plt.xlabel("Models")
    plt.ylabel("Precicions and Recall Score")
    plt.ylim(0.8, 1)
    plt.savefig('test_pos.png')

    results.plot(x = 'index', y = ["test neg precision", "test neg recall"], kind = "bar")
    plt.title("Precision and Recall of Different Models \n (Evaluated on Test; Respect to Negative")
    plt.xlabel("Models")
    plt.ylabel("Precicions and Recall Score")
    plt.ylim(0.8, 1)
    plt.savefig('test_neg.png')
        
    
    
    


if __name__ == "__main__":
    main(sys.argv)
