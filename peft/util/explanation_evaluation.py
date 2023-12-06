'''
This script takes the outputs of the models as seen in the peft/results/ folder and sets up human annotator evaluation interface 
to evalaute the model explanations on "factual" and "justifies"
    - The program asks the annotator to score model outputs
    - Results are stored in peft/results/human_evaluation/
'''

from accuracy_evaluation import *
import random


def get_evaluation_metric(filename):
    """
    Given a csv file containing query,model_output, and the two human evaluation metrics: factual,justifies
    outputs the overall average model explanation quality
    """
    with open(filename) as f:

        factual_num, factual_den = 0,0
        justifies_num, justifies_den = 0,0

        pos_factual_num, pos_factual_den = 0,0
        pos_justifies_num, pos_justifies_den = 0,0

        neg_factual_num, neg_factual_den = 0,0
        neg_justifies_num, neg_justifies_den = 0,0

        for line in f:
            factual = int(line.strip().split(',')[-3])
            justifies = int(line.strip().split(',')[-2])

            factual_num += factual
            justifies_num += justifies
            factual_den += 1
            justifies_den += 1

            if get_pos_or_neg(line.strip().split(',')[1]) == "neg":
                neg_factual_num += factual
                neg_factual_den += 1
                neg_justifies_num += justifies
                neg_justifies_den += 1
            else:
                pos_factual_num += factual
                pos_factual_den += 1
                pos_justifies_num += justifies
                pos_justifies_den += 1
                

        overall_score = "N/A" if (factual_den + justifies_den) == 0 else (factual_num + justifies_num)/(factual_den + justifies_den)
        overall_factual = "N/A" if factual_den == 0 else factual_num/factual_den
        overall_justifies = "N/A" if justifies_den == 0 else justifies_num/justifies_den

        pos_score = "N/A" if (pos_factual_den + pos_justifies_den) == 0 else (pos_factual_num + pos_justifies_num)/(pos_factual_den + pos_justifies_den)
        pos_factual = "N/A" if pos_factual_den == 0 else pos_factual_num/pos_factual_den
        pos_justifies = "N/A" if pos_justifies_den == 0 else pos_justifies_num/pos_justifies_den

        neg_score = "N/A" if (neg_factual_den + neg_justifies_den) == 0 else (neg_factual_num + neg_justifies_num)/(neg_factual_den + neg_justifies_den)
        neg_factual = "N/A" if neg_factual_den == 0 else neg_factual_num/neg_factual_den
        neg_justifies = "N/A" if neg_justifies_den == 0 else neg_justifies_num/neg_justifies_den
        
        print(f"{overall_score=} {overall_factual=} {overall_justifies=}")
        print(f"{pos_score=} {pos_factual=} {pos_justifies=}")
        print(f"{neg_score=} {neg_factual=} {neg_justifies=}")
        print()


def human_annotation_interface(input_filename, output_filename):
    """
    Sets up user interface for explanation annotation
        - Prints the query and the model output
        - Asks user to score the factuality and justification both as either 0 or 1
        - Stores the annotations in the output_filename
    """

    outputs = parse_results(input_filename, True)
    for i in range(len(outputs)):
        outputs[i].append(i)
    outputs_dict = dict()
    for i in outputs:
        outputs_dict[i[-1]] = i[:-1]
    
    with open(output_filename, 'r') as f:
        done = []
        for line in f:
            index = int(line.strip().split(',')[0])
            done.append(index)
    assert(len(done) == len(set(done)))

    print("Total number of samples:", len(outputs_dict.keys()))
    print("Number of samples evaluated:", len(done))
    print("Number of samples remaining:", len(outputs_dict.keys()) - len(done))
    print()


    todo = list(set(outputs_dict.keys()) - set(done))
    random.shuffle(todo)
    with open(output_filename, 'a') as f:
        count = 0
        for index in todo:
            query = outputs_dict[index][0]
            model_output = outputs_dict[index][1]

            print("Query:\t\t", query)
            print("Model output:\t", model_output)
            print()

            # ask user to evaluate factual
            factual = input("Factual: ")
            # ask user to evalaute justifies
            justifies = input("Justifies: ")

            if (factual == "0" or factual == "1") and (justifies == "0" or justifies == "1"):
                f.write(str(index) + ',' + query + ',' + model_output + ',' + str(factual) + ',' + str(justifies) + ',' + '\n')
                count+=1

            print()
            print()




def main():

    model_number = "6"
    input_filename = './results/model_outputs/model'+model_number+'.txt'
    output_filename = './results/human_evaluation/model'+model_number+'.csv'

    get_evaluation_metric(output_filename)
    human_annotation_interface(input_filename, output_filename)
    


if __name__ == "__main__":
    main()