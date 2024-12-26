import sys
import pandas as pd

def countTokens(text):
    token_counts = {}
    tokens = text.split(' ')
    for word in tokens:
        if not word in token_counts:
            token_counts[word] = 0
        token_counts[word] += 1
    return token_counts


def largest_counts(data): 

    neg_test_data = data[:12500]
    pos_train_data = data[25000:37500]
    pos_test_data = data[12500:25000]
    neg_train_data = data[37500:50000]

    def sortDic(dict):
        dict = dict(sorted(dict.items(), key = lambda x: -x[1]))
        return dict 

    #train pos 
    train_counts_pos_original = sortDic(countTokens(pos_train_data["review"].str.cat()))
    train_counts_pos_cleaned = sortDic(countTokens(
        pos_train_data["cleaned_review"].str.cat()))
    train_counts_pos_lowercased = sortDic(countTokens(
        pos_train_data["lowercased"].str.cat()))
    train_counts_pos_no_stop = sortDic(countTokens(
        pos_train_data["no stopwords"].str.cat()))
    train_counts_pos_lemmatized = sortDic(countTokens(
        pos_train_data["lemmatized"].str.cat()))
    
    #train neg
    train_counts_neg_original = sortDic(countTokens(neg_train_data["review"].str.cat()))
    train_counts_neg_cleaned = sortDic(countTokens(
        neg_train_data["cleaned_review"].str.cat()))
    train_counts_neg_lowercased = sortDic(countTokens(
        neg_train_data["lowercased"].str.cat()))
    train_counts_neg_no_stop = sortDic(countTokens(
        neg_train_data["no stopwords"].str.cat()))
    train_counts_neg_lemmatized = sortDic(countTokens(
        neg_train_data["lemmatized"].str.cat()))
    '''
    #test pos
    test_counts_pos_original = sortDic(countTokens(pos_test_data["review"].str.cat()))
    test_counts_pos_cleaned = sortDic(countTokens(
        pos_test_data["cleaned_review"].str.cat()))
    test_counts_pos_lowercased = sortDic(countTokens(
        pos_test_data["lowercased"].str.cat()))
    test_counts_pos_no_stop = sortDic(countTokens(
        pos_test_data["no stopwords"].str.cat()))
    test_counts_pos_lemmatized = sortDic(countTokens(
        pos_test_data["lemmatized"].str.cat()))
    
    #test neg
    test_counts_neg_original = sortDic(countTokens(neg_test_data["review"].str.cat()))
    test_counts_neg_cleaned = sortDic(countTokens(
        neg_test_data["cleaned_review"].str.cat()))
    test_counts_neg_lowercased = sortDic(countTokens(
        neg_test_data["lowercased"].str.cat()))
    test_counts_neg_no_stop = sortDic(countTokens(
        neg_test_data["no stopwords"].str.cat()))
    test_counts_neg_lemmatized = sortDic(countTokens(
        neg_test_data["lemmatized"].str.cat()))
    '''

    with open('counts.txt', 'w') as f:
        #train pos
        f.write('Original TRAIN POS reviews:\n')
        for k, v in list(train_counts_pos_original.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Cleaned TRAIN POS reviews:\n')
        for k, v in list(train_counts_pos_cleaned.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Lowercased TRAIN POS reviews:\n')
        for k, v in list(train_counts_pos_lowercased.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('No stopwords TRAIN POS reviews:\n')
        for k, v in list(train_counts_pos_no_stop.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Lemmatized TRAIN POS reviews:\n')
        for k, v in list(train_counts_pos_lemmatized.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))

        #train neg
        f.write('Original TRAIN NEG reviews:\n')
        for k, v in list(train_counts_neg_original.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Cleaned TRAIN NEG reviews:\n')
        for k, v in list(train_counts_neg_cleaned.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Lowercased TRAIN NEG reviews:\n')
        for k, v in list(train_counts_neg_lowercased.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('No stopwords TRAIN NEG reviews:\n')
        for k, v in list(train_counts_neg_no_stop.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Lemmatized TRAIN NEG reviews:\n')
        for k, v in list(train_counts_neg_lemmatized.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
            '''
        #test pos
        f.write('Original TEST POS reviews:\n')
        for k, v in list(test_counts_pos_original.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Cleaned TEST POS reviews:\n')
        for k, v in list(test_counts_pos_cleaned.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Lowercased TEST POS reviews:\n')
        for k, v in list(test_counts_pos_lowercased.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('No stopwords TEST POS reviews:\n')
        for k, v in list(test_counts_pos_no_stop.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Lemmatized TEST POS reviews:\n')
        for k, v in list(test_counts_pos_lemmatized.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))

        #test neg
        f.write('Original TEST NEG reviews:\n')
        for k, v in list(test_counts_neg_original.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Cleaned TEST NEG reviews:\n')
        for k, v in list(test_counts_neg_cleaned.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Lowercased TEST NEG reviews:\n')
        for k, v in list(test_counts_neg_lowercased.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('No stopwords TEST NEG reviews:\n')
        for k, v in list(test_counts_neg_no_stop.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))
        f.write('Lemmatized TEST NEG reviews:\n')
        for k, v in list(test_counts_neg_lemmatized.items())[:20]:
            f.write('{}\t{}\n'.format(k, v))    
        '''

def main(argv):
    data = pd.read_csv(argv[1], index_col=[0])
    # print(data.head())
    largest_counts(data)


if __name__ == "__main__":
    main(sys.argv)
