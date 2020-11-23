# This program determines the length of the longest repeated substring
# in a translation file.
#
# Run it with:
# python preprocess_scripts/find_repeated_substrings <path_to_fairseq_output_file>
#
import sys
from tqdm import tqdm

def longest_repeated_substring(string, substr_sizes=[1,2,3,4,5,6]):
    repeated_substr_counts = {}
    for i in range(len(string)):
        for s in substr_sizes:
            if i + s >= len(string):
                continue
            substr = string[i:i+s]
            substr_count = 0
            rem = string[i+s:]
            while len(rem) > s:
                if rem[:s] == substr:
                    substr_count+=1
                    rem = rem[s:]
                else:
                    break
            if substr in repeated_substr_counts:
                repeated_substr_counts[substr] = max(repeated_substr_counts[substr], substr_count)
            elif substr_count > 0:
                repeated_substr_counts[substr] = substr_count
    total_repeated_substring_counts = []
    if repeated_substr_counts == {}:
        return 0
    for s, c in repeated_substr_counts.items():
        total_repeated_substring_counts.append(c)
    return max(total_repeated_substring_counts)



if __name__ == "__main__":
    backtranslated_data = sys.argv[1]
    backtranslated_output = open(backtranslated_data).read().split("\n")
    outlines = []
    for line in tqdm(backtranslated_output): 
        if len(line) <= 6: 
            continue 
        line_type = line[0] 
        if line_type != "D": 
            continue 
        target_line = line.split('\t')[2] 
        outlines.append(target_line) 
    
    ss_lengths = [longest_repeated_substring(o) for o in outlines]
    for length in [5, 10, 15, 20, 25, 30]:
        num_substrings_above_length = len([l for l in ss_lengths if l > length])
        print(f"{round(100.0 * float(num_substrings_above_length) / len(ss_lengths), 2)} % of" +
        f" translations contain a repeated substring of {length} characters")
