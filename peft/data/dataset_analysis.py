import numpy as np
import matplotlib.pyplot as plt

'''
This file is a dataset analysis script
The results (summary statistics, etc.) from this script are used for the Dataset section in the paper 
'''

def get_generics_and_exemplars(filename):
    # Parses a csv and returns 2 lists: a list of generics and a list of exemplars
    with open(filename) as f:
        generics, exemplars = [], []
        for line in f:
            commaindex = line.strip().index(',')
            arr = [line[:commaindex], line[commaindex:]]
            if arr[0] == "generic":
                continue
            generic, exemplar = arr[0], arr[1]
            generics.append(generic)
            exemplars.append(exemplar)
    return generics, exemplars

def print_stats(arr, type):
    # Given an array of strings, print the average, min, and max character length and same with word count
    print(type)
    num_chars = np.array([len(i) for i in arr])
    char_min = np.min(num_chars)
    char_max = np.max(num_chars)
    char_avg = np.mean(num_chars)

    num_words = np.array([len(i.split(' ')) for i in arr])
    words_min = np.min(num_words)
    words_max = np.max(num_words)
    words_avg = np.mean(num_words)

    print(f"\t{char_avg=}\t{char_min=}\t{char_max=}")
    print(f"\t{words_avg=}\t{words_min=}\t{words_max=}")


def print_stats_of_file(filename):
    # Given a csv file, parse it, get the generics and exemplars, and print the summary statistics for it
    print(filename)
    generics, exemplars = get_generics_and_exemplars(filename)
    print_stats(generics, 'generics')
    print()
    print_stats(exemplars, 'exemplars')
    print()

def print_stats_of_both_files(filename1, filename2):
    print("Both files combined")
    generics1, exemplars1 = get_generics_and_exemplars(filename1)
    generics2, exemplars2 = get_generics_and_exemplars(filename2)
    generics = generics1 + generics2
    exemplars = exemplars1 + exemplars2
    print_stats(generics, 'generics')
    print()
    print_stats(exemplars, 'exemplars')
    print() 

print_stats_of_file('../all-exemplars-pruned/negative.csv')
print()
print()
print_stats_of_file('../all-exemplars-pruned/positive.csv')
print()
print()
print_stats_of_both_files('../all-exemplars-pruned/negative.csv', '../all-exemplars-pruned/positive.csv')




# plot the char length distributions of negative and positive exemplars as well as the total average of the whole dataset
neg_generics, neg_exemplars = get_generics_and_exemplars('../all-exemplars-pruned/negative.csv')
pos_generics, pos_exemplars = get_generics_and_exemplars('../all-exemplars-pruned/positive.csv')

neg_exemplar_char_length = np.array([len(i) for i in neg_exemplars])
pos_exemplar_char_length = np.array([len(i) for i in pos_exemplars])

plt.hist(pos_exemplar_char_length, alpha=0.6, label="Positive exemplars", bins=17)
plt.hist(neg_exemplar_char_length, alpha=0.6, label="Negative exemplars", bins=25)
plt.legend(loc='upper right')
plt.title('Exemplar character length')
plt.xlabel('Character length')
plt.ylabel('Count')
plt.savefig('./data/distributions.png')