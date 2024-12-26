def computeAccuracy(predictions, relevant_class):
    # The assert statement will notify you if the condition does not hold.
    assert len(predictions) == len(relevant_class)

    correctPredicts = 0
    totalPredicts = 0  
    
    for i in range(0,len(predictions)):
        
        if predictions[i] == relevant_class[i]:
            correctPredicts += 1

        totalPredicts += 1
        

    accuracy = correctPredicts / totalPredicts
    # return accuracy (the accuracy of the entire system)
    # and return mistakes (a list of the indices where we made mistakes)
    return accuracy


def computePrecisionRecall(predictions, gold_labels, relevant_class):
    assert len(predictions) == len(gold_labels)
    
    true_pos_count = 0
    false_pos_count = 0
    true_neg_count = 0 
    false_neg_count = 0 

    for i in range(0,len(predictions)):
        
        if relevant_class == "POSITIVE":
            
            if gold_labels[i] == 1 and predictions[i] == 1:
                true_pos_count += 1
            elif gold_labels[i] != 1 and predictions[i] == 1:
                false_pos_count += 1
            elif gold_labels[i] != 1 and predictions[i] != 1:
                true_neg_count += 1
            elif gold_labels[i] == 1 and predictions[i] != 1:
                false_neg_count += 1
        
        elif relevant_class == "NEGATIVE":
            
            if gold_labels[i] == 0 and predictions[i] == 0:
                true_pos_count += 1
            elif gold_labels[i] != 0 and predictions[i] == 0:
                false_pos_count += 1
            elif gold_labels[i] != 0 and predictions[i] != 0:
                true_neg_count += 1
            elif gold_labels[i] == 0 and predictions[i] != 0:
                false_neg_count += 1
    
    
    precision = true_pos_count / (true_pos_count + false_pos_count)
    recall = true_pos_count / (true_pos_count + false_neg_count)
    return (precision, recall)
