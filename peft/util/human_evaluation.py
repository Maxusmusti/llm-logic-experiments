'''
This script takes the outputs of the models as seen in the peft/results/ folder and sets up human annotator evaluation interface 
to evalaute the model explanations on "factual" and "justifies"
    - The program asks the annotator to score 20 model outputs at a time
    - Results are stored in peft/results/human_evaluation/
'''

from results_analysis import *
import random


def get_evaluation_metric(filename):
    """
    Given a csv file containing query,model_output, and the two human evaluation metrics: factual,justifies
    outputs the overall average model explanation quality
    """
    with open(filename) as f:
        factual_num, factual_den = 0,0
        justifies_num, justifies_den = 0,0
        for line in f:
            factual = int(line.strip().split(',')[-3])
            justifies = int(line.strip().split(',')[-2])

            factual_num += factual
            justifies_num += justifies
            factual_den += 1
            justifies_den += 1
        
        print("Overal:", (factual_num + justifies_num)/(factual_den + justifies_den))
        print("Factual:", factual_num/factual_den)
        print("Justifies:", justifies_num/justifies_den)


def human_annotation_interface(input_filename, output_filename):
    """
    Sets up user interface for explanation annotation
        - Prints the query and the model output
        - Asks user to score the factuality and justification both as either 0 or 1
        - Stores the annotations in the output_filename
        - Stops after every 20 samples to give the annotator a break
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


    todo = set(outputs_dict.keys()) - set(done)
    with open(output_filename, 'a') as f:
        count = 0
        for index in todo:
            query = outputs_dict[index][0]
            model_output = outputs_dict[index][1]

            print("Query:\t\t", query)
            print("Model output:\t", model_output)
            print()

            # ask user to evaluate factual
            factual = random.choice([0,1])

            # ask user to evalaute justifies
            justifies = random.choice([0,1])

            f.write(str(index) + ',' + query + ',' + model_output + ',' + str(factual) + ',' + str(justifies) + ',' + '\n')
            count+=1

            if count == 20:
                # every 20 samples, give the annotator a break
                break




def main():

    model_number = "4"
    input_filename = './results/model_outputs/model'+model_number+'.txt'
    output_filename = './results/human_evaluation/model'+model_number+'.csv'

    human_annotation_interface(input_filename, output_filename)
    get_evaluation_metric(output_filename)


if __name__ == "__main__":
    main()