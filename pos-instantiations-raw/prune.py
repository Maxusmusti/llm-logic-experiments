filtered = []
with open("instantiations.att.csv", 'r') as all_pos_file:
    for sample in all_pos_file:
        details = sample.strip().split(',')
        prob_valid = details[-1]
        prob_entail = details[-3]
        if prob_valid == "prob_valid":
            filtered.append("generic,exemplar")
            continue
        if float(prob_valid) > 0.95 and float(prob_entail) > 0.95:
            good = ','.join(details[:-3])
            filtered.append(good)
good_content = "\n".join(filtered)
with open("../instantiations-pruned/positive.csv", 'w') as pruned_file:
    pruned_file.write(good_content)
    pruned_file.write('\n')
