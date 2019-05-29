import numpy as np

r = ["love can always find a way", "love makes anything possible"]
c = ["the love can always do", "love can make anything possible"]

lambs= [0.5, 0.5, 0, 0]

rg = []
for ri in r:
    sri = ri.split()
    for n, l in enumerate(lambs):
        if l == 0:
            continue
        rg.append([tuple(sri[j:j+n+1]) for j in range(len(sri)-n)])

cg = []
for ci in c:
    sci = ci.split()
    for n, l in enumerate(lambs):
        if l == 0:
            continue
        cg.append([tuple(sci[j:j+n+1]) for j in range(len(sci)-n)])

