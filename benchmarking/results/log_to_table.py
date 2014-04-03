import sys
import re

# in which lines are the numbers you want in your column?
# arbirary order allowed (gives order of columns in table)
# can take lines several times -> several identical columns
# use illegal lines (e.g. 1000) to produce blank columns
read_rel_lines = [1, 1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9]
# which number to take from a line (0 for first number in line ...)
take_nr_nr =     [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
SPACES_FILL = 20
DELIMITER = ";"


if len(sys.argv) != 3:
    print "2 Arguments: input_file and output_file"
    exit(0)

in_filename = sys.argv[1]
out_filename = sys.argv[2]
in_file = open(in_filename,'r')
rel_line_count = 0
table = []

for line in in_file.readlines():
    rel_line_count += 1
    if line.find("=======") >= 0:
        rel_line_count = 0
        pos = line.find(".aag") - 1
        if pos < 0:
            print "Warning: header line must contain 'filename.aag'"
        name = ""
        while(pos >= 0 and line[pos] != ' '):
            name = line[pos] + name
            pos -= 1
        table.append([])
        table[-1].append(name)
        for pos in read_rel_lines:
            table[-1].append("")

    if rel_line_count in read_rel_lines:
        occ = re.compile('[0-9.]+').findall(line)
        if not occ:
            print "Warning: line %d contains no number" % rel_line_count
        else:
            for col_count in range(0,len(read_rel_lines)):
                if read_rel_lines[col_count] == rel_line_count and len(occ)>take_nr_nr[col_count]:
                    table[-1][col_count+1] = occ[take_nr_nr[col_count]]
in_file.close()

out_string = ""
for row in table:
    row_string = row[0]
    for elem_nr in range(1,len(row)):
        row_string += " " * ((elem_nr * SPACES_FILL) - len(row_string))
        row_string += DELIMITER + row[elem_nr]
    out_string += row_string + "\n"


out_file = open(out_filename,'a')
out_file.write(out_string)
out_file.close()
