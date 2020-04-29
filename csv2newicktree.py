import csv
from collections import defaultdict
from pprint import pprint

def tree(): return defaultdict(tree)

def tree_add(t, path):
  for node in path:
    t = t[node]

def pprint_tree(tree_instance):
    def dicts(t): return {k: dicts(t[k]) for k in t}
    pprint(dicts(tree_instance))

def csv_to_tree(input):
    t = tree()
    with open(input, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='\'')
        for row in reader:
            tree_add(t, row)
    return t

def tree_to_newick(root):
    items = []
    for k, v in root.items():
        s = ''
        if len(root[k].keys()) > 0:
            sub_tree = tree_to_newick(root[k])
            if sub_tree != '':
                s += '(' + sub_tree + ')'
        s += k
        items.append(s)
    return ','.join(items)

def csv_to_weightless_newick(input):
    t = csv_to_tree(input)
    #pprint_tree(t)
    return tree_to_newick(t)

input = '/home/stillsen/Documents/Uni/MasterArbeit/Source/Taxonomy_Analysis__Fungi_ZooScan/fungi_taxonomic_relationshit.csv'#["'phylum','class','order','family','genus','species'"]

print(csv_to_weightless_newick(input))