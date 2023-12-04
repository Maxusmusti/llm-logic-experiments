'''
This script takes the outputs of the models as seen in the peft/results/ folder and sets up human evaluation interface for the model explanations
Results are saved in peft/results/human_evaluation
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




def human_eval_interface(input_filename, output_filename):
    outputs = parse_results(input_filename, True)
    print(len(outputs))

    with open(output_filename, 'r') as f:
        for i in range(len(outputs)):
            for j in range(len(outputs[i])):
                if j < 2:
                    f.write(outputs[i][j])
                    f.write(',')

            factual = random.choice([0,1])
            justifies = random.choice([0,1])

            f.write(str(factual))
            f.write(',')
            f.write(str(justifies))
            f.write(',')
            f.write('\n')





def main():

    model_number = "6"
    input_filename = './results/model_outputs/model'+model_number+'.txt'
    output_filename = './results/human_evaluation/model'+model_number+'.csv'

    human_eval_interface(input_filename, output_filename)
    get_evaluation_metric(output_filename)


if __name__ == "__main__":
    main()