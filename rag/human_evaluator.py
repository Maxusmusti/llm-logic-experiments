import sys

to_eval = sys.argv[1]
if to_eval.endswith(".txt"):
    to_eval = to_eval[:-4]

with open(f"{to_eval}.txt", 'r') as data_file:
    data = data_file.read().split("\n\n-------------------\n\n")

fact_count = 0
just_count = 0
with open(f"human_eval_{to_eval}.txt", 'w') as eval_file:
    for i in range(len(data)):
        sample = data[i]
        if sample.startswith("INDEX: "):
            index = sample.split('\n')[0][7:]
            print(sample)

            factual = (input("Is the explanation factually correct? (y/n): ").lower() == 'y')
            just = (input("Does the explanation justify the answer? (y/n): ").lower() == 'y')

            if factual:
                fact_count += 1
            if just:
                just_count += 1

            eval_file.write("INDEX: " + index + "-" + str(i%2) + '\n')
            eval_file.write("FACTUAL: " + str(factual) + '\n')
            eval_file.write("JUSTIFIED: " + str(just))
            eval_file.write("\n\n-------------------\n\n")
            print('\n\n-------------------\n\n')

print("FACT COUNT: " + str(fact_count))
print("JUST COUNT: " + str(just_count))
