'''
This script takes the outputs of the models as seen in the peft/results/ folder and performs some further analysis on them:
    - 1-sentence truncation accuracy
'''

def get_tf_answer(answer):
    true_count = answer.lower().count("true")
    false_count = answer.lower().count("false")
    if true_count > false_count:
        return "true"
    elif true_count < false_count:
        return "false"
    else:
        return "tie"

def get_pos_or_neg(query):
    if query[:3] == "All" or query[:3] == "Not":
        return "neg"
    else:
        return "pos"

def parse_results(filepath, one_sentence_truncation):
    # Store the outputs into a matrix and format the model outputs that are multiple lines
    with open(filepath) as f:
        outputs = []
        for line in f:
            if line[:5] == "<QUER":
                outputs.append([line.strip()])
            elif line[:5] == "<MODE":
                outputs[-1].append(line.strip())
            elif line[:5] == "<EXPE":
                outputs[-1].append(line.strip())
            else:
                try:
                    outputs[-1][-1] = outputs[-1][-1] + line.strip()
                except:
                    pass

    # Remove the "<QUERY>", "<MODEL OUTPUT>", and "<EXPECTED OUTPUT>" parts
    for i in outputs:
        for index in range(len(i)):
            tab_index = i[index].index('\t')
            i[index] = i[index][tab_index:].strip()

    if one_sentence_truncation:
        # Truncate model outputs to be only 1 sentence
        for i in outputs:
            if '.' in i[1]:
                period_index = i[1].index('.')
                i[1] = i[1][:period_index+1]
    
    return outputs


def get_accuracy(outputs):
    correct, total, tie = 0, 0, 0
    neg_correct, neg_total, neg_tie = 0, 0, 0
    pos_correct, pos_total, pos_tie = 0, 0, 0
    for i in outputs:
        query = i[0]
        pred = i[1]
        original = i[2]

        expected_answer = get_tf_answer(original)
        model_answer = get_tf_answer(pred)

        if expected_answer == model_answer:
            correct += 1
        elif model_answer == "tie":
            tie += 1
        total += 1

        if get_pos_or_neg(query) == "neg":
            if expected_answer == model_answer:
                neg_correct += 1
            elif model_answer == "tie":
                neg_tie += 1
            neg_total += 1
        else:
            if expected_answer == model_answer:
                pos_correct += 1
            elif model_answer == "tie":
                pos_tie += 1
            pos_total += 1

    accuracy = correct / total
    incorrect = total - correct - tie

    neg_accuracy = neg_correct / neg_total
    neg_incorrect = neg_total - neg_correct - neg_tie

    pos_accuracy = pos_correct / pos_total
    pos_incorrect = pos_total - pos_correct - pos_tie

    print(f"{correct=} {tie=} {incorrect=} {total=} {accuracy=}")
    print(f"{neg_correct=} {neg_tie=} {neg_incorrect=} {neg_total=} {neg_accuracy=}")
    print(f"{pos_correct=} {pos_tie=} {pos_incorrect=} {pos_total=} {pos_accuracy=}")
    print()






def main():

    for i in range(4,7):
        print("======= Model "+str(i)+" accuracy results with 1-sentence truncation =======")
        outputs = parse_results('../results/model'+str(i)+'.txt', True)
        get_accuracy(outputs)


    # Explanation evaluation







if __name__ == "__main__":
    main()
