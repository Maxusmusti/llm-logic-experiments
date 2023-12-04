import sys

to_eval = sys.argv[1]
if to_eval.endswith(".txt"):
    to_eval = to_eval[:-4]

factual = 0
just = 0
with open(f"{to_eval}.txt", 'r') as data_file:
    data = data_file.read().split("\n\n-------------------\n\n")
    for sample in data:
        sample = sample.split('\n')
        if len(sample) > 1:
            factual += int(sample[1][9:] == "True")
            just += int(sample[2][11:] == "True")

print("FACTUAL: ", factual)
print("JUSTIFIED: ", just)
