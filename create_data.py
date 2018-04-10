# Script to sample first N lines of data

import sys

with open("chatbot_train/data/cornell-dialogs/movie_lines_cleaned.txt") as infile:
    head = [next(infile) for x in range(int(sys.argv[1]) * 1000)]

outfile = open("chatbot_train/data/cornell-dialogs/movie_lines_cleaned_" + sys.argv[1] +"k.txt", "w")

for line in head:
    outfile.write(line)

outfile.close()
