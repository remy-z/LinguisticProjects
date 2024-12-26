# Import the system module
import sys
import re
import string
import pathlib
import evaluation
import review_vector

POS_REVIEW = "POSITIVE"
NEG_REVIEW = "NEGATIVE"
NONE = "NONE"
POS = 'good'
NEG = 'bad'
reviewVecList = []
predictionList = []

def predictSimplistic(filename, label):
    

    # this used to be cleanFileContents
    # The below two lines open the file and read all the text from it
    # storing it into a variable called "text".

    with open(filename, 'r', encoding = 'utf8') as f:
        text = f.read()
        
    clean_text = text.translate(str.maketrans('', '', string.punctuation))

    # all space values into one space  
    clean_text = re.sub(r'\s+', ' ', clean_text)

    token_counts = {}
    
    token_list = clean_text.split()

    for word in token_list:
        if word not in token_counts:
            token_counts[word] = 0
       
        token_counts[word] += 1

    counts = token_counts
    pos_count = counts.get(POS, 0)
    neg_count = counts.get(NEG, 0)

    prediction = NONE
    if pos_count > neg_count:
        prediction = POS_REVIEW

    elif pos_count < neg_count:
        prediction = NEG_REVIEW

    elif pos_count == neg_count:
        prediction = NONE
        
    reviewVecList.append(review_vector.reviewVec(clean_text, label))

    predictionList.append(prediction)

def main(argv):
    
    dpos = pathlib.Path(sys.argv[1])  
    dneg = pathlib.Path(sys.argv[2])
     
    for filename in dpos.iterdir():
        if filename.suffix == '.txt':

            predictSimplistic(filename, POS_REVIEW)
            

    for filename in dneg.iterdir():
        if filename.suffix == '.txt':
            
            predictSimplistic(filename, NEG_REVIEW)
           
    # send values we found in predictSimplistic that were added to the lists reviewVecList and predictionList
    # to our evaluation functions to get the accuracy, precision and recall for our system 
    
    accuracy_and_mistake = evaluation.computeAccuracy(predictionList, reviewVecList)
    pre_and_re_wrt_POS = evaluation.computePrecisionRecall(predictionList, reviewVecList, POS_REVIEW)
    pre_and_re_wrt_NEG = evaluation.computePrecisionRecall(predictionList, reviewVecList, NEG_REVIEW)

    print(round(accuracy_and_mistake[0], 4))
    print(round(pre_and_re_wrt_POS[0], 4))
    print(round(pre_and_re_wrt_POS[1], 4))
    print(round(pre_and_re_wrt_NEG[0], 4))
    print(round(pre_and_re_wrt_NEG[1],4))
    
           
   




# The below code is needed so that this file can be used as a module.
# If we want to call our program from outside this window, in other words.
if __name__ == "__main__":
    main(sys.argv)
