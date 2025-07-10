import re

# Input and output file names
input_file = 'Mining\search_queries.txt'
output_file = 'Mining\queries_reformatted.txt'

# Read and process lines
with open(input_file, 'r') as infile:
    lines = infile.readlines()

reformatted_lines = []
for line in lines:
    line = line.strip()
    if not line:
        continue  # skip empty lines
    match = re.match(r'^site:([^\s]+)\s+(.*)', line)
    if match:
        domain = match.group(1)
        query = match.group(2)
        new_line = f"{query} site:{domain}"
        reformatted_lines.append(new_line)
    else:
        reformatted_lines.append(line)  # keep unchanged if no match

# Write to new file
with open(output_file, 'w') as outfile:
    for line in reformatted_lines:
        outfile.write(line + '\n')

print(f"Reformatted queries saved to '{output_file}'")
