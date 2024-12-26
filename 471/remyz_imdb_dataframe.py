import sys
import re
import string
import pathlib
import pandas as pd



# Constants:
POS = 1
NEG = 0


def createDataFrame(argv):
    new_filename = "my_imdb_dataframe.csv"

    data = []

    train_pos = pathlib.Path(sys.argv[1])  
    train_neg = pathlib.Path(sys.argv[2])
    test_pos = pathlib.Path(sys.argv[3])
    test_neg = pathlib.Path(sys.argv[4])
    
    for filename in train_pos.iterdir():
        if filename.suffix == '.txt':
            text = cleanFileContents(filename)
            data.append([filename, POS, "train", text])
    
    for filename in train_neg.iterdir():
        if filename.suffix == '.txt':
            text = cleanFileContents(filename)
            data.append([filename, NEG, "train", text])
    
    for filename in test_pos.iterdir():
        if filename.suffix == '.txt':
            text = cleanFileContents(filename)
            data.append([filename, POS, "test", text])
    
    for filename in test_neg.iterdir():
        if filename.suffix == '.txt':
            text = cleanFileContents(filename)
            data.append([filename, NEG, "test", text])
    
    print("hello")

    column_names = ["file", "label", "type", "review"]
    df = pd.DataFrame(data=data, columns=column_names)
    df.to_csv(new_filename)

def cleanFileContents(f):
    with open(f, 'r', encoding = 'utf8') as f:
        text = f.read()
    clean_text = text.translate(str.maketrans('', '', string.punctuation))
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text


def main(argv):
    createDataFrame(argv)


if __name__ == "__main__":
    main(sys.argv)
