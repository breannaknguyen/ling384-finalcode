from nltk import Tree
from collections import Counter
from pathlib import Path
import re
import csv

def normalize_label(label):
    return re.sub(r"[-=]\d+$", "", label)

treebank_dir = Path("PTB")
rule_counts = Counter()

# parse WSJ_0200 to WSJ_0299 
for i in range(200, 300):
    file_path = treebank_dir / f"wsj_{i:04}.mrg"
    if not file_path.exists():
        continue

    with file_path.open() as f:
        content = f.read()

    buffer = ""
    depth = 0
    for char in content:
        buffer += char
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                try:
                    tree = Tree.fromstring(buffer)
                    for prod in tree.productions():
                        lhs = normalize_label(str(prod.lhs()))
                        rhs = tuple(normalize_label(str(sym)) for sym in prod.rhs())
                        rule_counts[(lhs, rhs)] += 1
                except Exception as e:
                    print(f"Error parsing tree in {file_path.name}: {e}")
                buffer = ""

# export to CSV 
with open("all_rules.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["LHS", "RHS", "Count"])
    for (lhs, rhs), count in rule_counts.items():
        writer.writerow([lhs, " ".join(rhs), count])
