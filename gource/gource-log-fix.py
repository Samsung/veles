#!/usr/bin/env python3

import sys

if '__main__' == __name__:
    fr = open(sys.argv[1], 'r')
    fw = open(sys.argv[1] + ".fixed", 'w+')
    author=""
    date=""
    lines=[]
    for line in fr:
        if not line.startswith("|"):
            if line.startswith("user:"):
                author=line[5:].strip()
            elif line.strip() != "":
                date=int(line)
        else:
            lines.append((date, author + line))
    lines.sort(key=lambda line: line[0])
    for line in lines:
        fw.write(str(line[0]) + "|" + line[1])

