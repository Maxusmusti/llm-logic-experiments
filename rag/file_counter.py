import sys

to_eval = sys.argv[1]
if to_eval.endswith(".txt"):
    to_eval = to_eval[:-4]

win = 0
loss = 0
tie = 0
with open(f"{to_eval}.txt", 'r') as data_file:
    data = data_file.read().split("\n\n-------------------\n\n")
    for sample in data:
        result = sample.split('\n\n')[-1]
        if result.startswith("RESULT: "):
            result = result[8:]
            if result == "WIN":
                win += 1
            elif result == "LOSS":
                loss += 1
            elif result == "TIE":
                tie += 1
print("WINS: ", win)
print("LOSSES: ", loss)
print("TIES: ", tie)
